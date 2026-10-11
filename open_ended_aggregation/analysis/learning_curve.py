"""
How many LABELLED questions does the rank scorer actually need?

THE QUESTION THIS ANSWERS
  Every arm here is supervised, but the label APPETITE differs enormously:
  `count` fits one integer (the threshold), ours fits ~115 logistic coefficients
  plus a 256-cell pattern table plus the gold-count ridge. A reviewer -- and the
  supervisor, on 23 Aug -- will ask whether that is affordable, and whether
  coefficients fitted on one set of open-ended questions can possibly transfer to
  questions that are inherently different.

  That objection makes a PREDICTION. If per-question heterogeneity is the binding
  constraint, you need ever more training questions to cover the diversity and the
  curve keeps climbing. If the features are genuinely question-relative -- rank
  divided by the asserting agent's own list length, support percentile within the
  same question, log length of the silent agents -- then the coefficients describe
  how LLMs emit lists rather than anything about the subject matter, and the curve
  saturates early.

PROTOCOL
  A FIXED held-out test set (never trained on, identical at every N, so the points
  are comparable to each other). Training subsets of size N drawn at random from
  the remaining questions, R independent draws per N, everything refitted from
  scratch on each draw: pattern lookup, stack feature, logistic coefficients,
  threshold, gold-count ridge.

  This is deliberately NOT the 5-fold cross-fitting used for the headline. Cross-
  fitting answers "is it overfitting"; this answers "how much supervision does it
  cost". The N=max point should land near the cross-fitted headline modulo the
  smaller evaluation set.

ARMS
  count            threshold only        -- the honest supervision floor
  pattern          256-cell lookup + threshold
  rank             ~115 coefficients + threshold
  rank + budget    the above + the gold-count ridge   [the headline method]

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.learning_curve
  ./venv/bin/python -u -m open_ended_aggregation.analysis.learning_curve --draws 3
"""
import json, math, random, argparse, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern, GRID)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.paths import RESULTS

TEST_SEED = 0
SIZES = [25, 50, 100, 200, 350, 0]        # 0 = every remaining training question
ARMS = ["count", "pattern", "rank", "rank + budget"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--test", type=int, default=250, help="held-out questions")
    ap.add_argument("--test-seed", type=int, default=TEST_SEED,
                    help="which questions form the fixed held-out set")
    a = ap.parse_args()

    print("=" * 88)
    print("  LEARNING CURVE -- mean test F1 against the number of labelled "
          "training questions")
    print("=" * 88)

    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))

    spans = []
    for qi in range(nq):
        w = np.nonzero(rowq == qi)[0]
        spans.append((int(w[0]), int(w[-1]) + 1) if len(w) else (0, 0))
    C = lambda s: curves(qids, ngold, gis, spans, s)

    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for qi, q in enumerate(qids):
        w = np.nonzero(rowq == qi)[0]
        QF[qi, :4] = [1.0, math.log1p(len(w)) / 5.0,
                      (cnt[w].max() if len(w) else 0) / n, (cnt[w] >= 2).sum() / 20.0]
        QF[qi, 4 + tix.index(qtype[q])] = 1.0

    # ---- the fixed test set, drawn once and never trained on
    rng = random.Random(a.test_seed)
    order = list(range(nq)); rng.shuffle(order)
    te_q = np.array(sorted(order[:a.test]))
    pool = np.array(sorted(order[a.test:]))
    sizes = [s if s else len(pool) for s in SIZES]

    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct, {X.shape[1]} features")
    print(f"  {len(LEAN)} of them enter the rank scorer")
    print(f"  test set {len(te_q)} questions (fixed), training pool {len(pool)}")
    print(f"  sizes {sizes}, {a.draws} draws each\n", flush=True)

    def one_draw(tr_q):
        """Fit everything on tr_q alone; return mean test F1 per arm."""
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]
        out = {}

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        wl = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
        s_lean = X[:, LEAN] @ wl

        for arm, s in (("count", cnt), ("pattern", pat_p), ("rank", s_lean)):
            S, F = C(s)
            g = np.unique(s[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), g)
            got = apply_th(S, F, list(te_q), th)
            out[arm] = float(np.mean([got[qi] for qi in te_q]))

        p = 1.0 / (1.0 + np.exp(-np.clip(s_lean, -30, 30)))
        QG = np.zeros((nq, QF.shape[1] + 3))
        QG[:, :QF.shape[1]] = QF
        for qi in range(nq):
            w_ = np.nonzero(rowq == qi)[0]
            QG[qi, QF.shape[1]:] = [p[w_].sum() / 10.0,
                                    p[w_][cnt[w_] >= 2].sum() / 10.0,
                                    float(p[w_].max()) if len(w_) else 0.0]
        beta = fit_gold_ridge(QG[tr_q], np.log(np.maximum(ng_arr[tr_q], 1.0)))   # log-scale answer count
        Gh = np.maximum(np.exp(QG @ beta), 1.0)
        fa, _ = ev(adaptive_keep(p, rowq, nq, Gh), list(tr_q))
        out["rank + budget"] = float(fa[te_q].mean())
        return out

    curve = {}
    for N in sizes:
        draws = 1 if N >= len(pool) else a.draws
        got = {arm: [] for arm in ARMS}
        for d in range(draws):
            r = random.Random(1000 + N * 17 + d)
            tr_q = np.array(sorted(r.sample(list(pool), min(N, len(pool)))))
            res = one_draw(tr_q)
            for arm in ARMS:
                got[arm].append(100 * res[arm])
        curve[N] = {arm: (statistics.mean(v),
                          statistics.pstdev(v) if len(v) > 1 else 0.0)
                    for arm, v in got.items()}
        m = curve[N]
        print(f"    N={N:<5} " + "  ".join(
            f"{arm} {m[arm][0]:.2f}" + (f"+-{m[arm][1]:.2f}" if m[arm][1] else "")
            for arm in ARMS), flush=True)

    full = curve[sizes[-1]]
    print(f"\n  {'N train':>9}{'count':>9}{'pattern':>10}{'rank':>9}"
          f"{'rank+bud':>11}{'bud - count':>14}{'% of full gain':>17}")
    print("  " + "-" * 79)
    gain_full = full["rank + budget"][0] - full["count"][0]
    for N in sizes:
        m = curve[N]
        g = m["rank + budget"][0] - m["count"][0]
        print(f"  {N:>9}{m['count'][0]:>9.2f}{m['pattern'][0]:>10.2f}"
              f"{m['rank'][0]:>9.2f}{m['rank + budget'][0]:>11.2f}"
              f"{g:>+14.2f}{100*g/gain_full:>16.0f}%")

    print(f"\n  full-data gain over counting: {gain_full:+.2f} F1 "
          f"({100*gain_full/full['count'][0]:+.1f}%)")
    print("  +- is the spread over independent training draws at that size")

    out = RESULTS / f"learning_curve_test{a.test_seed}.json"
    json.dump({str(k): v for k, v in curve.items()}, open(out, "w"), indent=2)
    print(f"  wrote results/{out.name}")


if __name__ == "__main__":
    main()
