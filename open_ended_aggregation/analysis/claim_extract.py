"""
LLM claim extraction for long-form answers (pipeline steps 1-2 -> claim table).

WHY
  The scorer needs a claim table: which agents assert which fact, where, and
  who contradicts whom. Sentence-embedding clustering cannot build it on ASQA
  (docs/ASQA_MATCHING_HANDCHECK.md: 5/20 groups merge contradicting values, a
  third of true matches are missed). This replaces it with two LLM steps, and
  the model is never asked whether two agents' claims are "the same":

  A. SLOTS, one call per question. The model reads every answer (anonymised,
     order shuffled) and lists the distinct facts / sub-questions they address,
     as slots WITHOUT values. One shared slot list per question is what makes
     slots consistent across agents.
  B. FACTS, one call per answer. Given the slot list and ONE answer with
     numbered sentences, the model returns each factual statement as
     (slot id, value in canonical form, sentence number). Hedges and meta
     statements are skipped; a fact matching no slot gets a "NEW:" slot.

  Matching is then exact (methods/claim_table.py): agents AGREE when
  (slot, normalised value) is identical and CONTRADICT when the slot is the
  same and the value differs.

OUTPUT  results/claims_asqa.json   {qid: {question, slots, facts{agent: [...]},
                                         n_sent{agent: int}}}
SPEND   every call goes through paid.PaidCaller: cached, capped, --confirm only.

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.claim_extract --n 50            # estimate
  ./venv/bin/python -u -m open_ended_aggregation.analysis.claim_extract --n 50 --confirm  # spends
"""
import argparse, collections, json, random, re

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.synth_rungs import qrng
from open_ended_aggregation.paid import PaidCaller, NotConfirmed, CapReached
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
_SENT = re.compile(r"(?<=[.!?])\s+")

SLOT_ASK = """Several assistants answered the same question. List the distinct FACTS
or SUB-QUESTIONS their answers address, as slots WITHOUT values.

Rules:
- A slot names what is being stated, not what is claimed about it.
  Good: "Clay Matthews III career sacks". Bad: "Clay Matthews III has 91.5 sacks".
- If the question is ambiguous, each interpretation is its own slot.
- Two assistants giving DIFFERENT values for the same thing share ONE slot.
- Skip hedges and advice ("it depends", "check the official website").
- At most 15 slots, most important first.

Return JSON only: {"slots": [{"id": "S1", "slot": "..."}, ...]}
"""

FACT_ASK = """Extract every factual statement from ONE answer, mapped to the slots below.

Rules:
- One fact per (slot, value). Use the slot id; if no slot fits, use "NEW: <short slot name>".
- value: the shortest canonical form. Full official names for people, places and
  organisations; numbers as digits; years as YYYY; full dates as YYYY-MM-DD.
  No hedging words, no units unless needed.
- sentence: the number of the sentence the fact comes from.
- Skip hedges, advice and statements with no checkable content.

Return JSON only: {"facts": [{"slot": "S1", "value": "...", "sentence": 1}, ...]}
"""


LIST_ASK = """Below is ONE answer to a question that asks for a list of entities.
Rewrite every numbered item in canonical form: the full official name of the
entity (e.g. "MGM" -> "Metro-Goldwyn-Mayer", "NYC" -> "New York City").
Keep years, numbers and words that are part of the name or tell one entity from
another ("1969 World Series" stays "1969 World Series"); drop only side notes such
as "(film)" or "- won in 2001". Keep items that are not entities exactly as written.
Do not merge, drop or add items; return one canonical form per number.

Return JSON only: {"items": [{"i": 1, "canonical": "..."}, ...]}
"""


def sentences(t):
    return [s.strip() for s in _SENT.split(str(t or "").strip()) if s.strip()]


def load_asqa(n=None, gen="asqa_imb.jsonl"):
    """ASQA answers of the 6-agent pool; seeded sample of questions answered by all."""
    byq = collections.defaultdict(dict)
    for l in open(DATA / gen):
        o = json.loads(l)
        byq[o["qid"]].setdefault(o["model"], o)
    pool = sorted({m for d in byq.values() for m in d})
    items = {it["qid"]: it for it in A.load_items(100000)}
    qs = sorted(q for q, d in byq.items() if len(d) == len(pool) and q in items)
    random.Random(SEED).shuffle(qs)
    return byq, items, pool, (qs[:n] if n else qs)


def slot_msg(q, byq, items, pool):
    agents = pool[:]
    qrng(q, "agents").shuffle(agents)
    body = "\n\n".join(f"--- Assistant {j} ---\n{(byq[q][m]['resp'] or '').strip()}"
                       for j, m in enumerate(agents, 1))
    return SLOT_ASK + f"\nQuestion: {items[q]['question']}\n\n" + body


def fact_msg(q, m, byq, items, slots):
    sl = "\n".join(f"{s.get('id')}: {s.get('slot')}" for s in slots)
    sents = sentences(byq[q][m]["resp"])
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents, 1))
    return (FACT_ASK + f"\nQuestion: {items[q]['question']}\n\nSlots:\n{sl}\n\n"
            f"Answer (numbered sentences):\n{numbered}"), len(sents)


def parse_json(txt, key):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        v = json.loads(m.group(0))[key] if m else None
        return v if isinstance(v, list) else None
    except Exception:
        return None


def extract(qs, byq, items, pool, caller, log_every=10, workers=4):
    """Run steps A and B, `workers` questions at a time. Returns ({qid: record},
    failure counts)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(q):
        f = collections.Counter()
        slots = parse_json(caller(slot_msg(q, byq, items, pool), tag=f"{q}|slots")[0], "slots")
        if not slots:
            f["slots"] += 1
            return q, None, f
        rec = {"question": items[q]["question"], "slots": slots, "facts": {}, "n_sent": {}}
        for m in pool:
            msg, ns = fact_msg(q, m, byq, items, slots)
            facts = parse_json(caller(msg, tag=f"{q}|{m}")[0], "facts")
            if facts is None:
                f["facts"] += 1
                continue
            rec["facts"][m], rec["n_sent"][m] = facts, ns
        return q, rec, f

    out, fails = {}, collections.Counter()
    with ThreadPoolExecutor(workers) as ex:
        for qi, (q, rec, f) in enumerate(ex.map(one, qs), 1):
            fails.update(f)
            if rec:
                out[q] = rec
            if log_every and (qi % log_every == 0 or qi == len(qs)):
                print(f"    {qi}/{len(qs)}  new calls {caller.n_new}  cached {caller.n_cached}  "
                      f"spend ${caller.spent():.2f}  failures {dict(fails) or 0}", flush=True)
    return out, fails


# ------------------------------------------------------------------ lists (task 2b)
def alias_pairs_in(q, byq, items, pool):
    """Number of pairs of DISTINCT candidate strings that match the same gold
    answer (true aliases) -- the only pairs that can test matching recall."""
    from open_ended_aggregation.benchmarks import qampari as QA
    gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
             if str(a).strip()} for g in items[q]["gold"]]
    strings = {QA.norm(x) for m in pool for x in (byq[q][m].get("items") or []) if QA.norm(x)}
    by_g = collections.Counter(next((j for j, gs in enumerate(gold) if s_ in gs), None)
                               for s_ in strings)
    return sum(c * (c - 1) // 2 for j, c in by_g.items() if j is not None)


def load_qampari(n, agents=None, alias_rich=False):
    """QAMPARI answers (the 8-agent file), seeded sample of questions.
    alias_rich: keep only questions whose candidates contain true alias pairs.
    This selects on gold, which is fine for VALIDATING the extractor (a tool
    check) and must never be used for a method result: a random 20-question
    sample held a single alias pair."""
    from open_ended_aggregation.benchmarks import qampari as QA
    byq = collections.defaultdict(dict)
    for l in open(DATA / "qampari_asc800.jsonl"):
        o = json.loads(l)
        byq[o["qid"]].setdefault(o["model"], o)
    pool = sorted(agents or {m for d in byq.values() for m in d})
    items = {it["qid"]: it for it in QA.load_items(100000)}
    qs = sorted(q for q, d in byq.items() if set(pool) <= set(d) and q in items)
    random.Random(SEED).shuffle(qs)
    if alias_rich:
        qs = [q for q in qs if alias_pairs_in(q, byq, items, pool) > 0]
    return byq, items, pool, qs[:n]


def list_msg(q, m, byq, items):
    its = [x for x in (byq[q][m].get("items") or []) if str(x).strip()]
    body = "\n".join(f"{i}. {x}" for i, x in enumerate(its, 1))
    return LIST_ASK + f"\nQuestion: {items[q]['question']}\n\nAnswer:\n{body}", its


def extract_lists(qs, byq, items, pool, caller, log_every=10):
    """{qid: {agent: [[original, canonical], ...]}}; a parse failure keeps the
    originals (counted), so no agent silently loses its answer."""
    out, fails = {}, collections.Counter()
    for qi, q in enumerate(qs, 1):
        rec = {}
        for m in pool:
            msg, its = list_msg(q, m, byq, items)
            if not its:
                rec[m] = []
                continue
            got = parse_json(caller(msg, tag=f"{q}|{m}")[0], "items") or []
            canon = {}
            for g in got:
                try:
                    canon[int(g.get("i"))] = str(g.get("canonical") or "").strip()
                except (TypeError, ValueError):
                    pass
            if len(canon) < len(its):
                fails["missing_items"] += len(its) - len(canon)
            rec[m] = [[x, canon.get(i) or x] for i, x in enumerate(its, 1)]
        out[q] = rec
        if log_every and (qi % log_every == 0 or qi == len(qs)):
            print(f"    {qi}/{len(qs)}  new calls {caller.n_new}  cached {caller.n_cached}  "
                  f"spend ${caller.spent():.2f}  failures {dict(fails) or 0}", flush=True)
    return out, fails


def evaluate_lists(out, items):
    """Matching quality against gold aliases: LLM canonical forms vs the regex
    normaliser. A pair of DISTINCT original strings matching the same gold
    answer should match each other (recall); two matching different gold
    answers must not (false matches)."""
    from open_ended_aggregation.benchmarks import qampari as QA
    res = {}
    for label, keyf in (("regex norm", lambda o, c: QA.norm(o)),
                        ("LLM canonical", lambda o, c: QA.norm(c))):
        tp = fn = fp = 0
        for q, rec in out.items():
            gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                     if str(a).strip()} for g in items[q]["gold"]]
            strings = {}
            for pairs in rec.values():
                for o, c in pairs:
                    strings.setdefault(QA.norm(o), (o, c))
            gi = {s: next((j for j, gs in enumerate(gold) if s in gs), None) for s in strings}
            keyed = [(s, gi[s], keyf(*strings[s])) for s in strings if gi[s] is not None]
            for a in range(len(keyed)):
                for b in range(a + 1, len(keyed)):
                    same_gold, same_key = keyed[a][1] == keyed[b][1], keyed[a][2] == keyed[b][2]
                    tp += same_gold and same_key
                    fn += same_gold and not same_key
                    fp += (not same_gold) and same_key
        res[label] = dict(alias_recall=tp / max(1, tp + fn), false_matches=fp, alias_pairs=tp + fn)
        print(f"  {label:14s} alias pairs matched {tp}/{tp + fn} "
              f"({100*tp/max(1, tp + fn):.1f}%), false matches {fp}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="asqa", choices=["asqa", "qampari"])
    ap.add_argument("--alias-rich", action="store_true",
                    help="qampari only: questions containing true alias pairs (tool validation)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    if a.bench == "qampari":
        byq, items, pool, qs = load_qampari(a.n, alias_rich=a.alias_rich)
        print(f"  {len(qs)} questions, {sum(alias_pairs_in(q, byq, items, pool) for q in qs)} "
              f"true alias pairs to test")
        caller = PaidCaller(RESULTS / "claims_cache_qampari.jsonl", a.model, a.effort,
                            cap=a.max_cost, confirm=a.confirm, json_mode=True, max_out=6000)
        caller.estimate([list_msg(q, m, byq, items)[0] for q in qs for m in pool
                         if list_msg(q, m, byq, items)[1]], "list canonicalisation")
        if not a.confirm:
            print("  No --confirm: nothing was sent.")
            return
        try:
            out, fails = extract_lists(qs, byq, items, pool, caller)
        except (NotConfirmed, CapReached) as e:
            raise SystemExit(f"  STOPPED: {e}")
        fn = f"claims_qampari_{a.model.split('/')[-1]}.json"
        json.dump(out, open(RESULTS / fn, "w"), indent=1)
        print(f"  wrote results/{fn} ({len(out)} questions); "
              f"spend ${caller.spent():.2f}; failures {dict(fails) or 0}")
        evaluate_lists(out, items)
        return

    byq, items, pool, qs = load_asqa(a.n)
    caller = PaidCaller(RESULTS / "claims_cache_asqa.jsonl", a.model, a.effort,
                        cap=a.max_cost, confirm=a.confirm, json_mode=True, max_out=6000)
    # Step-B prompts depend on step-A slots; estimate them with a 600-char slot list.
    a_msgs = [slot_msg(q, byq, items, pool) for q in qs]
    b_msgs = [FACT_ASK + "x" * 600 + (byq[q][m]["resp"] or "") for q in qs for m in pool]
    caller.estimate(a_msgs, "step A (slots)")
    caller.estimate(b_msgs, "step B (facts, approx.)")
    if not a.confirm:
        print("  No --confirm: nothing was sent.")
        return
    try:
        out, fails = extract(qs, byq, items, pool, caller)
    except (NotConfirmed, CapReached) as e:
        raise SystemExit(f"  STOPPED: {e}")
    json.dump(out, open(RESULTS / "claims_asqa.json", "w"), indent=1)
    print(f"  wrote results/claims_asqa.json ({len(out)} questions); "
          f"spend ${caller.spent():.2f}; failures {dict(fails) or 0}")
    from open_ended_aggregation.methods.claim_table import report
    report(out, items, pool)


if __name__ == "__main__":
    main()
