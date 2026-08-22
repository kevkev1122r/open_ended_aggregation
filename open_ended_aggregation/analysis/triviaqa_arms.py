"""
TriviaQA: counting vs weighting, and the solo-ratio prediction.

WHY THIS IS A DIFFERENT SHAPE OF TASK
  QAMPARI and ASQA answers are SETS, so aggregation means filtering a candidate
  pool and the metric is set F1 / DR. TriviaQA answers are SINGLE strings, so
  aggregation means SELECTING one of the distinct answers the agents proposed.
  Two consequences worth stating explicitly in the paper:

    - rank+budget is UNDEFINED here. There is no within-list position (each
      agent emits one answer, so rank is always 0) and no keep-count to choose
      (you always return exactly 1). The mechanism needs a set-valued answer.
    - counting and weighting ARE defined, and this is the one dataset where
      published OW applies directly, which makes it the control.

WHAT IS BEING TESTED
  The solo-ratio screen predicts, from cached data alone and before any method
  is written, whether counting or weighting should win. Measured so far:

      QAMPARI  0.06   counting is already a near-perfect filter, weighting null
      ASQA     0.47   counting cannot filter, weighting is the whole game

  TriviaQA should sit near 1.0 -- a single strong agent's short answer is
  usually right, so a solo answer is barely worse than a supported one. The
  screen therefore predicts weighting should beat counting here, and clearly.
  If it does not, the screen is wrong and we need to know before it goes in a
  paper.

ARMS   best single | majority vote (counting) | weighted vote | oracle
  Weights and any tie-breaking are cross-fitted 5-fold over questions.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.triviaqa_arms
"""
import json, re, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
FOLDS = 5
_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def load(path, style=None):
    """qid -> {model: (normalised answer, was it correct)}"""
    recs = [json.loads(l) for l in open(path)]
    if style:
        recs = [r for r in recs if r.get("style") == style]
    byq = collections.defaultdict(dict)
    for r in recs:
        # the response may be a sentence; the cached `correct` flag is the
        # authority on grading, so the string is only used for AGREEMENT
        ans = r["resp"]
        if "Answer:" in str(ans):
            ans = str(ans).split("Answer:")[-1]
        k = norm(ans)
        if not k:
            continue
        byq[r["qid"]].setdefault(r["model"], (k, bool(r["correct"])))
    models = sorted({r["model"] for r in recs})
    qids = sorted(q for q, d in byq.items() if len(d) == len(models))
    return models, qids, byq


def solo_ratio(models, qids, byq):
    """P(answer correct | exactly 1 agent gives it) vs P(correct | >= 2)."""
    by_k = collections.defaultdict(lambda: [0, 0])
    for q in qids:
        grp = collections.defaultdict(list)
        for m in models:
            a, ok = byq[q][m]
            grp[a].append(ok)
        for a, oks in grp.items():
            k = len(oks)
            by_k[k][0] += any(oks); by_k[k][1] += 1
    return by_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="gen_full.jsonl")
    ap.add_argument("--style", default="natural")
    a = ap.parse_args()

    models, qids, byq = load(DATA / a.file, a.style if a.file == "gen_full.jsonl" else None)
    n = len(models)
    print("=" * 84)
    print(f"  TRIVIAQA -- {a.file}" + (f" (style={a.style})" if a.style else ""))
    print("=" * 84)
    print(f"\n  {len(qids)} questions, {n} agents\n")

    by_k = solo_ratio(models, qids, byq)
    tot = sum(v[1] for v in by_k.values())
    print(f"  {'k agents give it':>18}{'share':>9}{'P(correct)':>13}")
    for k in sorted(by_k):
        h, t = by_k[k]
        print(f"  {k:>18}{100*t/tot:8.1f}%{100*h/t:12.1f}%")
    p1 = by_k[1][0] / max(1, by_k[1][1])
    hi = sum(by_k[k][0] for k in by_k if k >= 2)
    ht = sum(by_k[k][1] for k in by_k if k >= 2)
    p2 = hi / max(1, ht)
    print(f"\n  SOLO RATIO = {100*p1:.1f}% / {100*p2:.1f}% = {p1/p2:.2f}"
          f"   (QAMPARI 0.06, ASQA 0.47)")

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}

    per = {arm: {} for arm in ("majority", "weighted")}
    for f in range(FOLDS):
        tr = [q for q in qids if fold[q] != f]
        te = [q for q in qids if fold[q] == f]
        acc = {m: (sum(byq[q][m][1] for q in tr) + 1) / (len(tr) + 2) for m in models}
        # OW's own weight: log-odds of the agent's accuracy
        ow = {m: math.log(acc[m] / (1 - acc[m])) for m in models}
        for q in te:
            grp = collections.defaultdict(list)
            for m in models:
                grp[byq[q][m][0]].append(m)
            # majority vote, ties broken by the strongest agent in the group
            best = max(grp.items(),
                       key=lambda kv: (len(kv[1]), max(acc[m] for m in kv[1])))
            per["majority"][q] = float(any(byq[q][m][1] for m in best[1]))
            bw = max(grp.items(), key=lambda kv: sum(ow[m] for m in kv[1]))
            per["weighted"][q] = float(any(byq[q][m][1] for m in bw[1]))

    singles = {m: [float(byq[q][m][1]) for q in qids] for m in models}
    bm = max(models, key=lambda m: statistics.mean(singles[m]))
    best_v = singles[bm]
    oracle = [float(any(byq[q][m][1] for m in models)) for q in qids]
    maj = [per["majority"][q] for q in qids]
    wgt = [per["weighted"][q] for q in qids]

    print(f"\n  per-agent accuracy")
    for m in sorted(models, key=lambda m: -statistics.mean(singles[m])):
        print(f"    {m:<40}{100*statistics.mean(singles[m]):6.2f}")

    print(f"\n  {'arm':<28}{'acc':>7}{'vs best single':>26}{'vs majority':>26}")
    print("  " + "-" * 87)

    def cell(v, ref):
        d, lo, hi_ = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi_:+5.2f}]{'*' if (lo>0 or hi_<0) else ' '}")

    rows = [(f"best single ({bm.split('/')[-1][:18]})", best_v),
            ("majority vote (counting)", maj),
            ("weighted vote (OW log-odds)", wgt),
            ("oracle (any agent right)", oracle)]
    res = {}
    for lbl, v in rows:
        res[lbl] = 100 * statistics.mean(v)
        c1 = f"{'(reference)':>26}" if v is best_v else cell(v, best_v)
        c2 = f"{'(reference)':>26}" if v is maj else cell(v, maj)
        print(f"  {lbl:<28}{res[lbl]:7.2f}{c1:>26}{c2:>26}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")
    print("  rank+budget is undefined here: single-answer task, no list position "
          "and no keep-count")

    json.dump(res, open(RESULTS / f"triviaqa_{a.file.split('.')[0]}.json", "w"),
              indent=2)
    print(f"  wrote results/triviaqa_{a.file.split('.')[0]}.json")


if __name__ == "__main__":
    main()
