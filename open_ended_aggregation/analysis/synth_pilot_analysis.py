"""
Analysis of the 80-question QAMPARI synthesis pilot (synth_rungs --tag n80).
Reads only saved files: no API calls.

  1. Paired bootstrap CIs on the arm differences that decide the pilot.
  2. Precision / recall / output size per arm.
  3. Does the LLM FOLLOW the numbers it is shown?  For every candidate in the
     p and pscramble prompts: the value shown next to it, and whether the LLM
     output it. If the LLM reads the table, inclusion should track the SHOWN
     value in both arms; if p carries information, only in the p arm should
     inclusion also track correctness.
  4. Matched size: at exactly the LLM's output size per question, does the
     scorer's own top-k beat the LLM's selection?  Separates "the LLM chose
     better" from "the LLM chose a different amount".

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.synth_pilot_analysis
"""
import json, collections, statistics

import numpy as np

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.benchmarks.qampari import norm, parse_list, score_set
from open_ended_aggregation.paths import RESULTS

ARMS = ("text", "p", "pscramble", "solo")


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    reps = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return d.mean(), np.quantile(reps, 0.025), np.quantile(reps, 0.975)


def table(prompt):
    """{candidate: shown value} parsed from the p / pscramble prompt."""
    out, on = {}, False
    for ln in prompt.splitlines():
        if ln.startswith("candidate | p(correct)"):
            on = True
            continue
        if on and " | " in ln:
            k, v = ln.rsplit(" | ", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def main():
    side = {json.loads(l)["qid"]: json.loads(l)
            for l in open(RESULTS / "synth_prompts_qampari_n80.jsonl")}
    cache = {}
    for l in open(RESULTS / "synth_cache_qampari.jsonl"):
        o = json.loads(l)
        cache[o["key"]] = o
    items = {it["qid"]: it for it in QA.load_items(100000)}
    qids = list(side)

    def gold_sets(q):
        return [{norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                 if str(a).strip()} for g in items[q]["gold"]]

    F = {a: [] for a in ARMS}; PR = {a: [] for a in ARMS}; RC = {a: [] for a in ARMS}
    N = {a: [] for a in ARMS}; SC = []; SCn = []
    follow = {a: {"shown": [], "out": [], "correct": []} for a in ("p", "pscramble")}
    novel = collections.Counter(); novel_ok = collections.Counter()
    matched = {a: [] for a in ("text", "p", "pscramble")}
    missing = 0
    for q in qids:
        s = side[q]
        gold = items[q]["gold"]
        gs = gold_sets(q)
        outs = {}
        for a in ARMS:
            c = cache.get(s["prompts"][a]["key"])
            if c is None:
                missing += 1
                break
            outs[a] = parse_list(c["text"])
        if len(outs) < len(ARMS):
            continue
        cands = set(s["p"])
        for a in ARMS:
            pr, rc, f1 = score_set(outs[a], gold)
            F[a].append(f1); PR[a].append(pr); RC[a].append(rc); N[a].append(len(outs[a]))
            for it in outs[a]:
                k = norm(it)
                if a != "solo" and k and k not in cands:
                    novel[a] += 1
                    novel_ok[a] += any(k in g for g in gs)
        keep = s["scorer_keep"]
        SC.append(score_set(keep, gold)[2]); SCn.append(len(keep))
        # 3. following the shown numbers
        for a in ("p", "pscramble"):
            shown = table(s["prompts"][a]["text"])
            chosen = {norm(x) for x in outs[a]}
            for k, v in shown.items():
                follow[a]["shown"].append(v)
                follow[a]["out"].append(1.0 if k in chosen else 0.0)
                follow[a]["correct"].append(1.0 if any(k in g for g in gs) else 0.0)
        # 4. matched size: scorer's top-k by TRUE p, k = the LLM's output size
        ranked = sorted(s["p"], key=lambda k: -s["p"][k])
        for a in matched:
            k = min(len(outs[a]), len(ranked))
            matched[a].append((score_set(outs[a], gold)[2], score_set(ranked[:k], gold)[2]))

    n = len(SC)
    print(f"\n  QAMPARI synthesis pilot: {n} paired questions ({missing} missing), "
          f"synthesizer gpt-5.4, zero-label p (new pool)")
    print("\n  1. arm means and paired differences (F1 x100, 95% bootstrap CI)")
    for a in ARMS:
        print(f"     {a:10s} F1 {100*np.mean(F[a]):6.2f}   P {100*np.mean(PR[a]):6.2f}"
              f"   R {100*np.mean(RC[a]):6.2f}   items {np.mean(N[a]):5.1f}")
    print(f"     {'scorer':10s} F1 {100*np.mean(SC):6.2f}{'':25s}items {np.mean(SCn):5.1f}")
    for lab, x, y in (("p - pscramble   (does p carry info to the LLM?)", F["p"], F["pscramble"]),
                      ("p - text        (does the table help at all?)", F["p"], F["text"]),
                      ("text - scorer   (plain synthesis vs no LLM)", F["text"], SC),
                      ("p - scorer      (the system vs no LLM)", F["p"], SC),
                      ("text - solo     (merging vs synthesizer alone)", F["text"], F["solo"])):
        d, lo, hi = boot(x, y)
        sig = "SIG" if lo > 0 or hi < 0 else "ns"
        print(f"     {lab:48s} {100*d:+6.2f}  [{100*lo:+6.2f}, {100*hi:+6.2f}]  {sig}")

    print("\n  3. does the LLM follow the numbers it is shown?  (candidate level)")
    for a in ("p", "pscramble"):
        sh = np.array(follow[a]["shown"]); o = np.array(follow[a]["out"]); c = np.array(follow[a]["correct"])
        hi_ = sh >= np.median(sh)
        print(f"     {a:10s} P(output | shown value high) {100*o[hi_].mean():5.1f}%   "
              f"| low {100*o[~hi_].mean():5.1f}%   corr(output, shown) {np.corrcoef(o, sh)[0,1]:+.2f}"
              f"   corr(output, correct) {np.corrcoef(o, c)[0,1]:+.2f}")

    print("\n  4. matched size: LLM selection vs the scorer's top-k at the LLM's own size")
    for a, pairs in matched.items():
        llm = [x for x, _ in pairs]; top = [y for _, y in pairs]
        d, lo, hi = boot(llm, top)
        print(f"     {a:10s} LLM {100*np.mean(llm):6.2f}  scorer top-k {100*np.mean(top):6.2f}"
              f"   LLM - scorer {100*d:+6.2f} [{100*lo:+6.2f}, {100*hi:+6.2f}]")

    print("\n  items the LLM added that no agent had asserted")
    for a in ("text", "p", "pscramble"):
        tot = sum(N[a])
        print(f"     {a:10s} {novel[a]:4d} of {tot} output items ({100*novel[a]/max(1,tot):.1f}%),"
              f" of which correct {novel_ok[a]}")


if __name__ == "__main__":
    main()
