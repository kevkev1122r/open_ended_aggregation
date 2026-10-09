"""
Rungs 1-3 of the evidence-to-synthesizer ladder, on cached generations.

THE QUESTION
  A synthesizer reading the n agent responses can, in principle, derive where a
  claim sat in its author's output and how much a silent agent wrote instead.
  Does explicitly extracting and annotating that evidence still help?

THE THREE ARMS
  1 text     the n responses, given separately and intact. The model may derive
             position and omission itself if it wants to.
  2 evidence the same responses plus a per-candidate evidence table (support,
             within-output position, graded omission).
  3 scramble identical to 2, except the evidence rows are permuted across
             candidates. Same tokens, same format, wrong assignment.

  4 p        the same responses plus ONE number per candidate: the probability
             it is correct, from the pool-invariant scorer fitted on QAMPARI
             only (analysis/pool_transfer.py). Zero-label on the target: no
             CWQ/WebQSP gold is used to produce it. This is the system.
  5 pscramble  identical to 4, p permuted across candidates. The control for 4.
  6 solo     the synthesizer answers the question ALONE, no responses. Without
             it, any gain of a strong synthesizer is confounded with its own
             knowledge of the answer.

  Also reported, no API call: `scorer` = the same p with the source-fitted
  per-question budget and no LLM at all. The LLM arms must beat this, or the
  LLM is decoration.

  ORDER. Assistant order and candidate order are randomised per question
  (seeded by qid), and the SAME permutation is used in every arm, so the arms
  stay paired. Sorted order hands one agent the first slot on every question,
  and position bias (guo2024serial) would then act as a hidden weight.

  Arm 3 is the control that matters. LLMs are unreliable at using numeric
  side-information and very good at appearing to. If 2 == 3 the model is not
  reading the table and any gain over 1 is formatting, not evidence.

SYNTHESIZER
  gpt-5.4, which is NOT in the CWQ pool -- a synthesizer drawn from the pool it
  is merging would have a self-preference confound sitting on the measurement.

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.synth_rungs \
      --bench cwq --n 40 --synth gpt-5.4
"""
import argparse
import collections
import json
import random
import statistics

import hashlib
import sys

from open_ended_aggregation.backends import azure, openrouter
from open_ended_aggregation.benchmarks.qampari import (norm, parse_list, score_set,
                                                       strip_special)
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0


MAX_OUT = 3000
ABORT_AFTER = 5          # consecutive failed calls before we stop and say so


def _backend(model):
    """Same routing convention as the generation runs: a '/' means OpenRouter."""
    return openrouter if "/" in model else azure


class RunAborted(RuntimeError):
    pass

ASK = ("Several assistants each answered the same question. Merge their answers "
       "into one final list of correct entities.\n"
       "Copy each entity exactly as it is written in the answers; do not "
       "rename, expand or translate it.\n"
       "Output ONLY the final list, one entity per line, no numbering, no "
       "commentary.\n")

EVIDENCE_NOTE = (
    "\nYou are also given an evidence table. For each candidate it records:\n"
    "  asserted   how many assistants listed it, out of the total\n"
    "  position   where the earliest asserter placed it in its own list, and "
    "how long that list was\n"
    "  silence    the longest list written by an assistant that did NOT list "
    "this candidate\n"
    "Use the table as evidence about which candidates are likely correct.\n")


P_NOTE = (
    "\nYou are also given, for each candidate, an estimated probability that it "
    "is correct. It comes from a statistical model fitted on other questions "
    "and is roughly calibrated: of candidates marked 0.30, about 30% are "
    "correct. Use it as evidence about which candidates are likely correct.\n")

SOLO_ASK = ("Answer the question with a list of correct entities.\n"
            "Output ONLY the list, one entity per line, no numbering, no "
            "commentary.\n")


# ---------------------------------------------------------------- spend safety
# Every paid response is cached on disk and reused on any rerun, so a crash, an
# abort or a rescoring never pays twice. The file is a paid cache: never delete
# it to "start clean".
def _ckey(synth, effort, msg):
    return hashlib.sha256(f"{synth}\x00{effort}\x00{msg}".encode()).hexdigest()


def load_cache(path):
    c = {}
    if path.exists():
        for ln in open(path):
            try:
                o = json.loads(ln)
                c[o["key"]] = o
            except Exception:
                pass
    return c


def qrng(q, salt=""):
    """Per-question RNG, stable across runs and processes (hash() is salted)."""
    return random.Random(int(hashlib.sha256((q + salt).encode()).hexdigest()[:12], 16))


def load(bench):
    if bench == "qampari":
        # Pilot option A: the 5 Azure agents' answers (the TARGET pool). The
        # scorer never sees these agents; see pool_transfer.zero_label_p_pool.
        from open_ended_aggregation.analysis.pool_transfer import AZURE
        from open_ended_aggregation.benchmarks import qampari as QA
        by = collections.defaultdict(dict)
        for ln in open(DATA / "qampari_asc800.jsonl", errors="ignore"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o["model"] in AZURE:
                by[o["qid"]].setdefault(o["model"], o)   # first wins, as the scorer
        return by, {it["qid"]: it for it in QA.load_items(100000)}
    return _load_kg(bench)


def _load_kg(bench):
    gen = {"cwq": "cwq_gen", "qampari": "qampari_gen",
           "webqsp": "webqsp_gen"}[bench]
    by = collections.defaultdict(dict)
    for ln in open(DATA / f"{gen}.jsonl", errors="ignore"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        by[o["qid"]][o["model"]] = o
    items = {}
    for src in ("cwq_items.json", "webqsp_items.json"):
        try:
            for it in json.load(open(DATA / src)):
                items[it["qid"]] = it
        except Exception:
            pass
    return by, items


def build(rows):
    """Per-candidate evidence from the cached generations of one question."""
    models = sorted(rows)
    lens = {m: len(rows[m].get("items") or []) for m in models}
    pos, seen = {}, collections.defaultdict(set)
    for m in models:
        for i, it in enumerate(rows[m].get("items") or []):
            k = norm(it)
            if not k:
                continue
            seen[k].add(m)
            pos.setdefault(k, {}).setdefault(m, i)
    ev = {}
    for k, ms in seen.items():
        silent = [lens[m] for m in models if m not in ms]
        first_m = min(pos[k], key=lambda m: pos[k][m])
        ev[k] = dict(n=len(ms), of=len(models),
                     rank=pos[k][first_m] + 1, listlen=max(1, lens[first_m]),
                     silence=max(silent) if silent else 0)
    return ev, seen


def render(rows, ev, arm, q, P=None):
    """One prompt body. Every permutation is drawn from qrng(q, ...), so it is
    identical across arms for the same question."""
    agents = sorted(rows)
    qrng(q, "agents").shuffle(agents)
    parts = []
    for j, m in enumerate(agents, 1):
        # strip leaked chat tokens (<|START_TEXT|> etc.); scoring already does
        parts.append(f"--- Assistant {j} ---\n{strip_special(rows[m].get('resp') or '').strip()}")
    body = "\n\n".join(parts)
    if arm == "text":
        return body
    keys = sorted(ev)
    qrng(q, "cands").shuffle(keys)
    if arm in ("p", "pscramble"):
        vals = [P.get(k, 0.0) for k in keys]
        if arm == "pscramble":
            qrng(q, "pscramble").shuffle(vals)
        tbl = ["candidate | p(correct)"] + [f"{k} | {v:.2f}" for k, v in zip(keys, vals)]
        return body + P_NOTE + "\n" + "\n".join(tbl)
    vals = [ev[k] for k in keys]
    if arm == "scramble":
        vals = vals[:]
        qrng(q, "scramble").shuffle(vals)      # same rows, wrong candidates
    tbl = ["candidate | asserted | position | silence"]
    for k, v in zip(keys, vals):
        tbl.append(f"{k} | {v['n']} of {v['of']} | item {v['rank']} of "
                   f"{v['listlen']} | {v['silence']}")
    return body + EVIDENCE_NOTE + "\n" + "\n".join(tbl)


def list_messages(q, arm, rows, ev, question, Pq, Kq):
    """(system, user, info) for one arm. Existing arms keep byte-identical
    prompts, so their paid cache entries stay valid."""
    from open_ended_aggregation.methods import baselines as B
    from open_ended_aggregation.methods import composer as C
    agents = sorted(rows)
    qrng(q, "agents").shuffle(agents)                     # same order render() uses
    if arm == "solo":
        return None, SOLO_ASK + "\nQuestion: " + question, {}
    if arm == "usc":
        resp = [strip_special(rows[m].get("resp") or "").strip() for m in agents]
        return None, B.usc_prompt(question, resp), {"agents": agents}
    if arm == "moa":
        resp = [strip_special(rows[m].get("resp") or "").strip() for m in agents]
        sy, us = B.moa_messages(SOLO_ASK + "\nQuestion: " + question, resp)
        return sy, us, {}
    if arm == "band":
        acc, und, _ = C.band_split(Pq, C.list_cut(Pq, Kq))
        return None, C.list_band_prompt(question, render(rows, ev, "text", q, Pq),
                                        acc, und), {"acc": acc, "und": und}
    return None, ASK + "\nQuestion: " + question + "\n\n" + render(rows, ev, arm, q, Pq), {}


def msg_key(synth, effort, system, user):
    if system is None:
        return _ckey(synth, effort, user)                 # unchanged for existing arms
    return _ckey(synth, effort, "SYSTEM\x00" + system + "\x00USER\x00" + user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cwq")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--synth", default="gpt-5.4")
    ap.add_argument("--arms", default="text,p,pscramble,solo",
                    help="core arms by default; add evidence,scramble explicitly")
    ap.add_argument("--effort", default="low",
                    help="reasoning effort sent to OpenRouter (none|minimal|low|medium|high)")
    ap.add_argument("--max-cost", type=float, default=1.0,
                    help="hard cap in USD for NEW calls this run; aborts before exceeding")
    ap.add_argument("--confirm", action="store_true",
                    help="required to make paid calls; without it only the estimate prints")
    ap.add_argument("--tag", default="",
                    help="suffix for the results file, e.g. smoke")
    ap.add_argument("--p-source", default="qampari",
                    help="dataset the zero-label scorer is fitted on")
    ap.add_argument("--dry-run", action="store_true",
                    help="render every prompt, print one and a size estimate; no API calls")
    a = ap.parse_args()

    by, items = load(a.bench)
    full = max(len(v) for v in by.values())
    BK = _backend(a.synth)
    arms = a.arms.split(",")
    # zero-label p and the no-LLM scorer reference, computed BEFORE sampling so
    # the question sample is drawn only from questions that have them
    if a.bench == "qampari":
        from open_ended_aggregation.analysis.pool_transfer import zero_label_p_pool
        P, K, test_q = zero_label_p_pool()
        print("  zero-label: scorer fitted on the gpt-5.4 agents, TRAIN-half questions;"
              " applied to the Azure agents, TEST-half questions")
    else:
        from open_ended_aggregation.analysis.pool_transfer import zero_label_p
        P, K = zero_label_p(a.bench, source=a.p_source)
    qs = sorted(q for q, v in by.items() if len(v) == full and q in items and q in P)
    rng = random.Random(SEED)
    rng.shuffle(qs)
    qs = qs[:a.n]
    print(f"  {a.bench}: {len(qs)} questions x {full} agents, synthesizer {a.synth}")

    # ---- cost estimate from real prices (prompt chars/4; output = the cap)
    msgs = {}
    for q in qs:
        ev_, _ = build(by[q])
        for arm in arms:
            msgs[(q, arm)] = list_messages(q, arm, by[q], ev_, items[q]["question"],
                                           P[q], K.get(q, set()))
    cache_path = RESULTS / f"synth_cache_{a.bench}.jsonl"
    cache = load_cache(cache_path)
    todo = [k for k, (sy, us, _) in msgs.items()
            if msg_key(a.synth, a.effort, sy, us) not in cache]
    in_tok = sum(len(msgs[k][1]) + len(msgs[k][0] or "") for k in todo) / 4
    if BK is openrouter:
        pin, pout = openrouter.prices().get(a.synth, (None, None))
        if pin is None:
            sys.exit(f"  {a.synth} is not a known OpenRouter model id -- nothing run.")
        lo = in_tok * pin + len(todo) * 300 * pout
        hi = in_tok * pin + len(todo) * MAX_OUT * pout
        print(f"  COST ESTIMATE  {len(todo)} new calls ({len(msgs) - len(todo)} cached), "
              f"~{in_tok/1e6:.2f}M input tokens\n"
              f"    ${lo:.2f} if ~300 output tokens per call, "
              f"${hi:.2f} worst case ({MAX_OUT} per call); cap ${a.max_cost:.2f}, "
              f"effort={a.effort}")
    if not a.dry_run and not a.confirm:
        print("  No --confirm: nothing was sent. Re-run with --confirm to spend.")
        return

    # Exact prompts + the scorer's p and keep-set for every question, written
    # BEFORE any call, so later analysis never depends on regenerating them.
    if not a.dry_run:
        side = RESULTS / f"synth_prompts_{a.bench}{'_' + a.tag if a.tag else ''}.jsonl"
        with open(side, "w") as fh:
            for q in qs:
                fh.write(json.dumps({"qid": q, "p": P[q], "scorer_keep": sorted(K.get(q, ())),
                                     "prompts": {arm: {"key": msg_key(a.synth, a.effort,
                                                                      msgs[(q, arm)][0],
                                                                      msgs[(q, arm)][1]),
                                                       "system": msgs[(q, arm)][0],
                                                       "text": msgs[(q, arm)][1],
                                                       "info": msgs[(q, arm)][2]}
                                                 for arm in arms}})
                         + "\n")
        print(f"  prompts + scorer outputs saved to {side.name}")

    if a.dry_run:
        chars = collections.Counter()
        for q in qs:
            ev, _ = build(by[q])
            for arm in arms:
                body = ("" if arm == "solo" else render(by[q], ev, arm, q, P[q]))
                chars[arm] += len(body) + len(items[q]["question"]) + 300
        q0 = qs[0]
        ev0, _ = build(by[q0])
        print("\n  ---- example prompt, arm p ----\n")
        print(ASK + "\nQuestion: " + items[q0]["question"] + "\n\n"
              + render(by[q0], ev0, "p", q0, P[q0])[:4000])
        tot = sum(chars.values()) / 4
        print(f"\n  DRY RUN: {len(qs)} q x {len(arms)} arms, ~{tot/1e6:.2f}M input "
              f"tokens (chars/4) + up to {MAX_OUT} output tokens per call")
        for arm in arms:
            print(f"    {arm:10s} ~{chars[arm]/4/len(qs):8.0f} input tokens / question")
        return
    out = {k: [] for k in arms}
    ref = {"best_single": collections.defaultdict(list), "union": [], "scorer": []}
    detail, errors, consec, last_err = [], collections.Counter(), 0, None
    n_cached = 0

    for qi, q in enumerate(qs, 1):
        rows, gold = by[q], items[q]["gold"]
        ev, seen = build(rows)
        rec = {"qid": q, "n_cand": len(ev)}
        got = {}
        for arm in arms:
            system, msg, info = msgs[(q, arm)]
            key = msg_key(a.synth, a.effort, system, msg)
            if key in cache:
                txt, meta = cache[key]["text"], cache[key]["meta"]
                n_cached += 1
            else:
                if BK.usage["cost"] >= a.max_cost:
                    raise RunAborted(f"spend cap ${a.max_cost:.2f} reached "
                                     f"(${BK.usage['cost']:.2f}); all paid responses "
                                     f"are cached in {cache_path.name}")
                kw = ({"extra": {"reasoning": {"effort": a.effort}}}
                      if BK is openrouter else {})
                txt, meta = BK.chat(a.synth, system, msg, MAX_OUT, **kw)
                if txt:
                    rec_c = {"key": key, "qid": q, "arm": arm, "synth": a.synth,
                             "effort": a.effort, "text": txt, "meta": meta}
                    cache[key] = rec_c
                    with open(cache_path, "a") as fh:
                        fh.write(json.dumps(rec_c) + "\n")
            rec[arm + "_rt"] = (meta or {}).get("reasoning", 0)
            rec[arm + "_finish"] = (meta or {}).get("finish")
            if not txt:
                # A FAILED CALL IS NOT A ZERO. Scoring it as an empty prediction
                # is indistinguishable from a real bad answer, which is how a
                # dead run masquerades as a result -- the v1 failure, again.
                errors[arm] += 1
                consec += 1
                last_err = (meta or {}).get("error", "empty response")
                if consec >= ABORT_AFTER:
                    raise RunAborted(
                        f"{consec} consecutive failed calls; last error:\n"
                        f"      {str(last_err)[:300]}")
                continue
            consec = 0
            if arm == "usc":
                from open_ended_aggregation.methods.baselines import usc_choice
                j = usc_choice(txt, len(info["agents"]))
                if j is None:                      # unparseable selection: a failure, not a zero
                    errors["usc_parse"] += 1
                    continue
                pred = rows[info["agents"][j]].get("items") or []
            elif arm == "band":
                from open_ended_aggregation.methods.composer import list_band_output
                pred = list_band_output(txt, info["acc"], info["und"])
            else:
                pred = parse_list(txt)
            pr_, rc_, f1_ = score_set(pred, gold)
            got[arm] = f1_
            rec[arm + "_prec"], rec[arm + "_rec"] = round(100 * pr_, 2), round(100 * rc_, 2)
            # Output SIZE per arm. On these benchmarks most of any rule's gain is
            # how many items it keeps (cut_rules.py), so an LLM arm that "wins"
            # by listing more or fewer items has to be visible as such.
            rec[arm + "_n"] = len(pred)

        # Only keep a question when EVERY arm returned, so the arms stay paired.
        if len(got) == len(arms):
            for arm in arms:
                out[arm].append(got[arm])
                rec[arm] = round(100 * got[arm], 2)
            detail.append(rec)
            ref["union"].append(score_set(list(seen), gold)[2])
            ref["scorer"].append(score_set(list(K.get(q, ())), gold)[2])
            rec["ref_scorer"] = round(100 * ref["scorer"][-1], 2)
            rec["ref_scorer_n"] = len(K.get(q, ()))
            rec["ref_union"] = round(100 * ref["union"][-1], 2)
            rec["ref_count2"] = round(100 * score_set(
                [k for k in ev if ev[k]["n"] >= 2], gold)[2], 2)
            for m, r in rows.items():
                ref["best_single"][m].append(r.get("f1") or 0.0)
        if qi % 10 == 0 or qi == len(qs):
            msg = "  ".join(f"{k} {100*statistics.mean(v):5.2f}"
                            for k, v in out.items() if v)
            print(f"    {qi:4d}/{len(qs)}  scored {len(detail):4d}  "
                  f"failed {sum(errors.values()):3d}  {msg}   "
                  f"spend ${BK.usage['cost']:.2f}", flush=True)

    n_ok = len(detail)
    print(f"\n  {a.bench}: requested {len(qs)}, SCORED {n_ok}, "
          f"failed calls {sum(errors.values())} {dict(errors) or ''}")
    if n_ok == 0:
        raise RunAborted("nothing scored -- no result to report")
    if n_ok < 0.9 * len(qs):
        print(f"    *** PARTIAL RUN: {len(qs)-n_ok} questions dropped. The "
              f"survivors are not a random sample -- do not read these means "
              f"as a result. ***")
    print(f"  synthesizer={a.synth}")
    bs = max(ref["best_single"].values(), key=lambda v: statistics.mean(v))
    print(f"    {'best single agent':22s} {100*statistics.mean(bs):6.2f}")
    print(f"    {'union of all agents':22s} {100*statistics.mean(ref['union']):6.2f}")
    print(f"    {'scorer alone, no LLM':22s} {100*statistics.mean(ref['scorer']):6.2f}")
    for arm in arms:
        sizes = [r[arm + "_n"] for r in detail if arm + "_n" in r]
        print(f"    {'synth: ' + arm:22s} {100*statistics.mean(out[arm]):6.2f}"
              f"   mean items {statistics.mean(sizes) if sizes else 0:5.1f}")
    ks = [len(K.get(r["qid"], ())) for r in detail]
    print(f"    {'(scorer alone items)':22s}         mean items {statistics.mean(ks):5.1f}")
    if "evidence" in out and "scramble" in out:
        d = 100 * (statistics.mean(out["evidence"]) - statistics.mean(out["scramble"]))
        print(f"\n    evidence - scramble = {d:+.2f}   "
              f"(the control: near zero means the table is being ignored)")
    if "p" in out and "pscramble" in out:
        d = 100 * (statistics.mean(out["p"]) - statistics.mean(out["pscramble"]))
        print(f"    p - pscramble       = {d:+.2f}   (same control, for the p arm)")
    rts = [r[k] for r in detail for k in r if k.endswith("_rt") and r[k]]
    cut = sum(1 for r in detail for k in r if k.endswith("_finish") and r[k] == "length")
    print(f"    reasoning tokens / call: mean {statistics.mean(rts) if rts else 0:.0f}, "
          f"max {max(rts) if rts else 0};  truncated calls {cut};  "
          f"cached (free) calls {n_cached}")
    print(f"    total spend this run ${BK.usage['cost']:.2f}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"bench": a.bench, "n": len(qs), "synth": a.synth,
               "means": {k: 100 * statistics.mean(v) for k, v in out.items()},
               "best_single": 100 * statistics.mean(bs),
               "union": 100 * statistics.mean(ref["union"]),
               "scorer_alone": 100 * statistics.mean(ref["scorer"]),
               "n_scored": n_ok, "n_requested": len(qs),
               "failed_calls": dict(errors), "complete": n_ok >= 0.9 * len(qs),
               "cost": BK.usage["cost"], "per_question": detail},
              open(RESULTS / f"synth_rungs_{a.bench}{'_' + a.tag if a.tag else ''}.json",
                   "w"), indent=1)
    print(f"    wrote results/synth_rungs_{a.bench}{'_' + a.tag if a.tag else ''}.json")


if __name__ == "__main__":
    import sys
    try:
        main()
    except RunAborted as e:
        sys.exit(f"\n  ABORTED: {e}\n  No result written. Nothing here is "
                 f"usable until the cause is fixed.")
