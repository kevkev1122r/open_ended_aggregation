"""
Which property of a model pool predicts whether reliability weighting helps?

DESIGN
  The same 8 agents were run on BOTH QAMPARI and QUEST. For each benchmark we
  form every 5-of-8 subset (C(8,5) = 56 pools), evaluate the same two arms on
  identical questions, and record pool-composition statistics alongside the
  outcome. That turns one result per benchmark into 56 paired data points and
  lets us ask whether composition PREDICTS the outcome rather than guessing.

WHY IT MATTERS
  QAMPARI says weighting adds nothing (+0.08). QUEST says weighting is the only
  thing that works (+1.47). Those are not contradictory if some property of the
  pool governs it. QAMPARI's sweep found weighting gain tracks best-mean(rest)
  at r=+0.581 -- but QAMPARI's ABSOLUTE gap is larger than QUEST's (0.093 vs
  0.077) while its RATIO is smaller (4.82x vs 7.65x), so the absolute statistic
  cannot explain the flip. This tests several candidate statistics on both
  benchmarks at once; a predictor that only works on one is not a mechanism.

ARMS (per subset, identical to compare_methods)
  MA-count      count filter over models, theta = 2
  MA-count+OW   same filter, weights = cross-fitted per-model precision,
                theta chosen EXACTLY by enumerating subset-sum breakpoints
                (a threshold on subset sums has at most 2^5 distinct behaviours)

Usage:  ./venv/bin/python subset_sweep_both.py
"""
import sys, os, json, math, itertools, statistics, collections, random

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.benchmarks import quest as QU
from open_ended_aggregation.analysis.verify_qampari import bootstrap

POOL8 = ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5",
         "MAI-Thinking-1", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "grok-4.3"]


def load_qampari():
    recs = [json.loads(l) for l in open(f"{HERE}/data/qampari_asc800.jsonl")]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    qids = sorted(q for q, d in byq.items() if set(POOL8) <= set(d))
    items = {it["qid"]: it for it in QA.load_items(100000) if it["qid"] in set(qids)}
    qids = [q for q in qids if q in items]
    gold = {q: items[q]["gold"] for q in qids}
    return byq, qids, gold, (lambda pred, g: QA.score_set(pred, g)[2]), QA.norm


def load_quest():
    recs = [json.loads(l) for l in open(QU.GEN_PATH)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    items = {it["qid"]: it for it in QU.load_items(400)}
    qids = sorted(q for q, d in byq.items() if set(POOL8) <= set(d) and q in items)
    gold = {q: items[q]["gold"] for q in qids}
    return byq, qids, gold, (lambda pred, g: QU.score_set(pred, g)[2]), QU.norm


def sweep(name, byq, qids, gold, f1of, norm):
    KEY = {(q, m): [k for k in dict.fromkeys(norm(i) for i in byq[q][m]["items"]) if k]
           for q in qids for m in POOL8}
    REP = {}
    for q in qids:
        for m in POOL8:
            for i in byq[q][m]["items"]:
                REP.setdefault((q, norm(i)), i)
    prec = {m: statistics.mean(byq[q][m]["prec"] for q in qids) for m in POOL8}
    single = {m: [f1of(byq[q][m]["items"], gold[q]) for q in qids] for m in POOL8}

    def evaluate(pool, w, theta):
        out = []
        for q in qids:
            acc = collections.defaultdict(float)
            for m in pool:
                for k in KEY[(q, m)]:
                    acc[k] += w[m]
            out.append(f1of([REP[(q, k)] for k, v in acc.items() if v >= theta - 1e-12],
                            gold[q]))
        return out

    rows = []
    for pool in itertools.combinations(POOL8, 5):
        w = {m: prec[m] for m in pool}
        subs = {frozenset(c) for r in range(6) for c in itertools.combinations(pool, r)}
        best_arm = None
        for t in sorted({round(sum(w[m] for m in s), 9) for s in subs}):
            v = evaluate(pool, w, t)
            mu = statistics.mean(v)
            if best_arm is None or mu > best_arm[0]:
                best_arm = (mu, v)
        ow = best_arm[1]
        cnt = evaluate(pool, {m: 1.0 for m in pool}, 2)
        bm = max(pool, key=lambda m: statistics.mean(single[m]))
        p = sorted((prec[m] for m in pool), reverse=True)
        rest = statistics.mean(p[1:])
        d1, l1, h1 = bootstrap(ow, cnt)
        d2, l2, h2 = bootstrap(ow, single[bm])
        rows.append(dict(
            pool=pool,
            gap=p[0] - rest,                       # absolute dominance
            ratio_rest=p[0] / max(1e-9, rest),     # relative dominance
            ratio_min=p[0] / max(1e-9, p[-1]),
            cv=statistics.pstdev(p) / max(1e-9, statistics.mean(p)),
            std=statistics.pstdev(p),
            nweak=sum(1 for x in p if x < 0.5 * p[0]),
            ow_cnt=d1, sig1=(l1 > 0 or h1 < 0),
            ow_best=d2, sig2=(l2 > 0 or h2 < 0)))
    print(f"\n  {name}: {len(qids)} questions, {len(rows)} subsets, "
          f"mean weighting gain {statistics.mean(r['ow_cnt'] for r in rows):+.2f}, "
          f"significant in {sum(r['sig1'] for r in rows)}/{len(rows)}")
    return rows


def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


if __name__ == "__main__":
    RA = sweep("QAMPARI", *load_qampari())
    RQ = sweep("QUEST", *load_quest())

    STATS = [("best - mean(rest)  ABSOLUTE", "gap"),
             ("best / mean(rest)  RELATIVE", "ratio_rest"),
             ("best / weakest", "ratio_min"),
             ("coefficient of variation", "cv"),
             ("std of precision", "std"),
             ("# models below half the best", "nweak")]

    print(f"\n  correlation with WEIGHTING GAIN (MA-count+OW  -  MA-count)")
    print(f"  {'predictor':<34}{'QAMPARI':>10}{'QUEST':>10}{'consistent?':>14}")
    print("  " + "-" * 70)
    for lbl, k in STATS:
        ca = corr([r[k] for r in RA], [r["ow_cnt"] for r in RA])
        cq = corr([r[k] for r in RQ], [r["ow_cnt"] for r in RQ])
        ok = "YES" if (ca > 0.25 and cq > 0.25) or (ca < -0.25 and cq < -0.25) else ""
        print(f"  {lbl:<34}{ca:>+10.3f}{cq:>+10.3f}{ok:>14}")

    print(f"\n  correlation with AGGREGATION GAIN (MA-count+OW  -  best single)")
    print(f"  {'predictor':<34}{'QAMPARI':>10}{'QUEST':>10}")
    print("  " + "-" * 56)
    for lbl, k in STATS:
        print(f"  {lbl:<34}"
              f"{corr([r[k] for r in RA], [r['ow_best'] for r in RA]):>+10.3f}"
              f"{corr([r[k] for r in RQ], [r['ow_best'] for r in RQ]):>+10.3f}")

    print(f"\n  benchmark-level values (all 8 agents)")
    for nm, R in [("QAMPARI", RA), ("QUEST", RQ)]:
        print(f"    {nm:<10} mean gap {statistics.mean(r['gap'] for r in R):.3f}   "
              f"mean ratio_rest {statistics.mean(r['ratio_rest'] for r in R):.2f}x   "
              f"mean CV {statistics.mean(r['cv'] for r in R):.3f}")

    json.dump({"qampari": [{**r, "pool": list(r["pool"])} for r in RA],
               "quest": [{**r, "pool": list(r["pool"])} for r in RQ]},
              open(f"{HERE}/results/subset_sweep_both.json", "w"), indent=1)
    print("\n  wrote results/subset_sweep_both.json")
