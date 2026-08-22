"""
SECONDARY ANALYSIS: what predicts how much our method gains over the best single agent?

THE ASK (21 Aug meeting)
  "Let's say we have a set of models with these accuracies and some relation
   between how their performance varies at a per-question level. Can we use that
   to predict how much improvement we can get by using this technique on them?"

  If the answer is yes, the paper stops being "our method gains X on benchmark Y"
  and becomes "here is the rule for when this method is worth deploying" -- which
  is a claim about aggregation rather than about our benchmarks.

WHAT IS REGRESSED
  target   gain of rank+budget over the BEST SINGLE agent in that ensemble
           (also reported over counting, since those are the paper's two tables)
  units    all 210 QAMPARI ensembles of size 3-6

PREDICTORS  -- three families, all computable BEFORE running the method

  capability      mean / max / min single-agent F1 in the ensemble
  heterogeneity   dominance (best / mean of the rest), coefficient of variation,
                  spread (max - min)
  complementarity how much the non-best agents know that the best one does not:
                    - union recall minus best-agent recall
                    - mean per-question SD of member F1 (do they vary together?)
                    - mean pairwise Jaccard of the claim sets they assert
                    - the ensemble's solo ratio, P(correct|k=1)/P(correct|k>=2)

  The third family is the interesting one. Capability and heterogeneity are what
  a reviewer would guess at; complementarity is the hypothesis that what matters
  is whether the weaker agents contribute anything the best agent misses.

HONESTY
  R^2 is reported both in-sample and leave-one-out cross-validated. Ensembles
  overlap (they are subsets of the same 8 agents) so the LOO figure is optimistic
  too; the ranking of predictors is the durable part, not the absolute R^2.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.gain_predictors
"""
import json, math, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import load, make_eval
from open_ended_aggregation.paths import RESULTS


def main():
    rows = json.load(open(RESULTS / "rank_crossover.json"))
    models, qids, cand, ngold, nlist, qtype = load()
    n, nq = len(models), len(qids)
    mi = {m: i for i, m in enumerate(models)}

    # per (question, agent) F1, and the claim sets each agent asserts
    Z, rowq, gis = [], [], []
    for qi, q in enumerate(qids):
        for ms, gi, rk in cand[q]:
            z = np.zeros(n, dtype=bool)
            for m in ms:
                z[mi[m]] = True
            Z.append(z); rowq.append(qi); gis.append(gi)
    Z = np.array(Z); rowq = np.array(rowq)
    ev = make_eval(qids, ngold, rowq, gis)

    singleF = np.zeros((n, nq))
    for j in range(n):
        fa, _ = ev(Z[:, j], list(range(nq)))
        singleF[j] = fa
    ok = np.array([g is not None for g in gis])
    ngold_arr = np.array([max(1, ngold[q]) for q in qids], dtype=float)

    def union_recall(S):
        m = Z[:, S].any(1) & ok
        u = np.unique(rowq[m].astype(np.int64) * 1000 +
                      np.array([gis[i] for i in np.nonzero(m)[0]]))
        c = np.bincount((u // 1000).astype(int), minlength=nq).astype(float)
        return float((c / ngold_arr).mean())

    def agent_recall(j):
        m = Z[:, j] & ok
        u = np.unique(rowq[m].astype(np.int64) * 1000 +
                      np.array([gis[i] for i in np.nonzero(m)[0]]))
        c = np.bincount((u // 1000).astype(int), minlength=nq).astype(float)
        return float((c / ngold_arr).mean())

    arec = [agent_recall(j) for j in range(n)]

    # mean pairwise Jaccard of asserted claim sets
    J = np.zeros((n, n))
    for a, b in itertools.combinations(range(n), 2):
        inter = (Z[:, a] & Z[:, b]).sum()
        uni = (Z[:, a] | Z[:, b]).sum()
        J[a, b] = J[b, a] = inter / max(1, uni)

    def solo_ratio(S):
        c = Z[:, S].sum(1)
        s1 = c == 1; s2 = c >= 2
        p1 = ok[s1].mean() if s1.any() else 0.0
        p2 = ok[s2].mean() if s2.any() else 1e-9
        return p1 / max(1e-9, p2)

    NAMES = ["size", "mean F1", "max F1", "min F1", "dominance", "CV of F1",
             "spread", "union recall - best", "per-q SD of member F1",
             "mean pairwise Jaccard", "solo ratio"]
    X, y_best, y_cnt = [], [], []
    for r in rows:
        S = [mi[m] for m in r["ens"]]
        f = [float(singleF[j].mean()) for j in S]
        f_sorted = sorted(f, reverse=True)
        bj = S[int(np.argmax(f))]
        perq_sd = float(np.mean(np.std(singleF[S], axis=0)))
        jac = float(np.mean([J[a, b] for a, b in itertools.combinations(S, 2)]))
        X.append([
            r["k"], statistics.mean(f), max(f), min(f),
            f_sorted[0] / max(1e-9, statistics.mean(f_sorted[1:])),
            statistics.pstdev(f) / max(1e-9, statistics.mean(f)),
            max(f) - min(f),
            union_recall(S) - arec[bj],
            perq_sd, jac, solo_ratio(S),
        ])
        y_best.append(r["rank+budget"] - r["best"])
        y_cnt.append(r["rank+budget"] - r["count"])
    X = np.array(X); y_best = np.array(y_best); y_cnt = np.array(y_cnt)

    print("=" * 86)
    print("  WHAT PREDICTS THE GAIN?  210 QAMPARI ensembles, sizes 3-6")
    print("=" * 86)
    print(f"\n  gain over BEST SINGLE   mean {y_best.mean():+.2f}  "
          f"sd {y_best.std():.2f}  range [{y_best.min():+.2f}, {y_best.max():+.2f}]")
    print(f"  gain over COUNTING      mean {y_cnt.mean():+.2f}  "
          f"sd {y_cnt.std():.2f}  range [{y_cnt.min():+.2f}, {y_cnt.max():+.2f}]")

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = math.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    print(f"\n  UNIVARIATE CORRELATION with the gain")
    print(f"  {'predictor':<26}{'vs best single':>16}{'vs counting':>14}")
    print("  " + "-" * 56)
    order = sorted(range(len(NAMES)), key=lambda i: -abs(corr(X[:, i], y_best)))
    for i in order:
        print(f"  {NAMES[i]:<26}{corr(X[:, i], y_best):>+16.3f}"
              f"{corr(X[:, i], y_cnt):>+14.3f}")

    def fit_r2(Xs, y):
        A = np.column_stack([np.ones(len(Xs)), Xs])
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = A @ w
        ss = ((y - pred) ** 2).sum(); tt = ((y - y.mean()) ** 2).sum()
        # leave-one-out
        loo = []
        for i in range(len(y)):
            m = np.ones(len(y), dtype=bool); m[i] = False
            wi = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
            loo.append(A[i] @ wi)
        loo = np.array(loo)
        ss2 = ((y - loo) ** 2).sum()
        return 1 - ss / tt, 1 - ss2 / tt, w

    print(f"\n  MULTIVARIATE FIT (standardised predictors)")
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    for lbl, y in (("gain vs best single", y_best), ("gain vs counting", y_cnt)):
        r2, r2loo, w = fit_r2(Xs, y)
        print(f"\n    {lbl}:  R2 = {r2:.3f}   leave-one-out R2 = {r2loo:.3f}")
        idx = sorted(range(len(NAMES)), key=lambda i: -abs(w[i + 1]))
        for i in idx[:5]:
            print(f"      {NAMES[i]:<26}{w[i+1]:>+8.3f}")

    print(f"\n  PARSIMONIOUS MODELS (vs best single)")
    for combo in ([0], [4], [7], [0, 7], [4, 7], [0, 4, 7], [0, 4, 7, 8]):
        r2, r2loo, _ = fit_r2(Xs[:, combo], y_best)
        print(f"    {'+'.join(NAMES[i] for i in combo):<52}"
              f"R2 {r2:.3f}  LOO {r2loo:.3f}")

    json.dump({"names": NAMES,
               "corr_vs_best": [corr(X[:, i], y_best) for i in range(len(NAMES))],
               "corr_vs_count": [corr(X[:, i], y_cnt) for i in range(len(NAMES))]},
              open(RESULTS / "gain_predictors.json", "w"), indent=2)
    print("\n  wrote results/gain_predictors.json")


if __name__ == "__main__":
    main()
