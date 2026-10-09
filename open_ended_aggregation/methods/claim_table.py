"""
Prose claim table (pipeline step 2) from extracted facts.

For one question and a set of agents, every distinct (slot, value) is a claim:

  supporters   agents asserting exactly this value, with the sentence it came
               from (their position signal)
  contradict   agents asserting a DIFFERENT value in the same slot
  silent       agents asserting nothing in the slot

Lists never had "contradict": an agent that did not list an item was simply
silent. In prose the two are different evidence and are kept apart.

Slots may be MULTI-VALUED (an ambiguous question can have several right
answers), so "contradict" means "answered this slot, but not with this value".
An agent that states two values in one slot supports both and contradicts
neither.

Restricting to a subset of agents rebuilds the table from their facts only, so
a pool never sees evidence from agents outside it.
"""
import collections
import datetime
import re
import statistics

from open_ended_aggregation.benchmarks import asqa as A

_MONTH = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b")


def date_key(s):
    """ISO key for a FULL date (day, month and year), else None. The extractor
    writes dates as YYYY-MM-DD while ASQA gold says "April 13, 2018"; without a
    shared key every correct date fact would be labelled wrong."""
    s0 = str(s or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s0)
    if m:
        try:
            return "date:" + datetime.date(*map(int, m.groups())).isoformat()
        except ValueError:
            return None
    if _MONTH.search(s0) and re.search(r"\b\d{4}\b", s0) and re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", s0):
        try:
            from dateutil import parser
            d = parser.parse(s0, default=datetime.datetime(1, 1, 1), fuzzy=True)
            if d.year > 1:
                return "date:" + d.date().isoformat()
        except (ValueError, OverflowError):
            return None
    return None


def value_key(v):
    return date_key(v) or A.norm(v)


def build(rec, agents):
    """rec: one question from claims_asqa.json. Returns (claims, slot_values)."""
    by_sv = collections.defaultdict(dict)            # (slot, value) -> {agent: sentence}
    display = {}                                     # (slot, value) -> first raw wording
    slot_agents = collections.defaultdict(set)       # slot -> agents with any fact in it
    for m in agents:
        for f in rec["facts"].get(m, []) or []:
            s, v = str(f.get("slot", "")).strip(), value_key(f.get("value", ""))
            if not s or not v:
                continue
            try:
                sent = int(f.get("sentence", 0) or 0)
            except (TypeError, ValueError):
                sent = 0
            display.setdefault((s, v), str(f.get("value", "")).strip())
            prev = by_sv[(s, v)].get(m)
            by_sv[(s, v)][m] = sent if prev is None else min(prev, sent)
            slot_agents[s].add(m)
    slot_values = collections.defaultdict(dict)      # slot -> {value: n supporters}
    for (s, v), sup in by_sv.items():
        slot_values[s][v] = len(sup)
    claims = []
    for (s, v), sup in by_sv.items():
        claims.append(dict(
            slot=s, value=v, display=display.get((s, v), v), supporters=dict(sup),
            contradict=sorted(slot_agents[s] - set(sup)),
            silent=sorted(set(agents) - slot_agents[s]),
            n_sent={m: rec["n_sent"].get(m, 0) for m in agents}))
    return claims, slot_values


def gold_aliases(item):
    return [value_key(x) for s in item["short_sets"] for x in s if str(x).strip()]


def is_correct(value, aliases):
    """Partial label: value matches an ASQA short-answer alias. Context facts
    have no gold and count as not-correct -- a lower bound on precision."""
    v = f" {value} "
    return any(g and (g == value or f" {g} " in v) for g in aliases)


def report(out, items, agents):
    fpa, spq, new, contra, agree_k = [], [], 0, [], collections.Counter()
    ok_k = collections.defaultdict(list)
    for q, rec in out.items():
        spq.append(len(rec["slots"]))
        for m in agents:
            fs = rec["facts"].get(m, []) or []
            fpa.append(len(fs))
            new += sum(1 for f in fs if str(f.get("slot", "")).startswith("NEW"))
        claims, slot_values = build(rec, agents)
        al = gold_aliases(items[q])
        for c in claims:
            k = len(c["supporters"])
            agree_k[k] += 1
            ok_k[k].append(is_correct(c["value"], al))
        contra += [len(v) > 1 for v in slot_values.values()]
    n = sum(agree_k.values()) or 1
    print(f"\n  facts per answer {statistics.mean(fpa):.1f}; slots per question "
          f"{statistics.mean(spq):.1f}; facts placed in NEW slots {new}")
    print(f"  slots with >1 distinct value (contradiction or paraphrase): "
          f"{100*statistics.mean(contra) if contra else 0:.0f}%")
    print("  claims by number of agents asserting them, and share matching an ASQA short answer:")
    for k in sorted(agree_k):
        print(f"    {k} agents: {agree_k[k]:5d} ({100*agree_k[k]/n:4.1f}%)  "
              f"match gold {100*statistics.mean(ok_k[k]):5.1f}%")
