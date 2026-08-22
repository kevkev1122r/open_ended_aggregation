"""
Stack every base scorer that survived, bag the GBM, then budget. The final arm.

WHAT SURVIVED, AND WHAT DID NOT
  Roughly fifty rules have now been cross-fitted on QAMPARI. The ones that
  carry independent signal:

    pattern        best possible who-said-what rule            +0.25   ns
    RRF(K=10)      parameter-free IR list fusion               +0.84   *
    2D count x rank  P(correct | count, min-rank bucket)       +0.71   *
    nb-rank        rank-conditioned Naive Bayes with omission  +1.34   *
    LEAN           logistic: rank, omission strength, stack    +1.44   *
    GBM            same features, boosted                      +0.43   ns  (but best AP)
    + per-question budget on top of any of them                +0.9 to +1.2

  and the ones that are dead, each tested and each a usable negative:
  error-correlation weighting (lambda=0 in 5/5 folds), pairwise co-assertion
  (+0.05), rank agreement (+0.07), question-domain weights (+0.00), Borda /
  CombSUM / wBorda (all BELOW counting), isotonic calibration (already
  calibrated), within-question normalisation (-0.9 to -2.6), learned k (-3.7),
  and alias merging (bad merges outnumber good at every threshold).

WHY STACK
  The survivors fail differently. `pattern` is a nonparametric estimate that is
  well calibrated across questions but blind outside the support pattern. `GBM`
  ranks best within a question (AP 0.397 -> 0.418) but is the worst calibrated
  across them. `RRF` and the 2D lookup are low-variance and assumption-light.
  A meta-logistic over all of them, fitted on a slice of TRAIN the base models
  did not see, is the standard way to buy the union of their strengths.

  The GBM is also BAGGED over three seeds, because a single boosted fit on a
  3.5%-positive target has visible seed variance.

HONESTY
  Base scorers are fitted on the inner 85% of the training questions; the meta
  model and GBM early stopping use the remaining 15%; thresholds are chosen on
  the full training half; the test fold is touched by none of it.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v8 [--seed N]
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import (
    bucket, NB_BUCKETS, fit_nb_rank, score_nb, fit_gold_ridge, adaptive_keep)
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.weighting_v6 import rank_tensors, fusion_scores, lookup2d
from open_ended_aggregation.analysis.gbm import GBM, bin_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS


def ap_score(p, y):
    o = np.argsort(-p); ys = y[o]
    return float(((np.cumsum(ys) / np.arange(1, len(ys) + 1)) * ys).sum()
                 / max(1.0, ys.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bags", type=int, default=3)
    ap.add_argument("--trees", type=int, default=500)
    a = ap.parse_args()

    print("=" * 96)
    print("  WEIGHTING v8 -- stacked ensemble of every surviving scorer, then budget")
    print("=" * 96)
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    X, y, rowq, pats, gis, RKb, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    RK, LL, _ = rank_tensors(models, qids, cand, nlist)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    Z = RK >= 0
    B = bucket(np.where(Z, RK, 0).astype(int))
    minr = np.where(Z, RK, np.inf).min(1)
    mb = bucket(np.where(np.isfinite(minr), minr, 0).astype(int))
    code = (Z * (1 << np.arange(n))).sum(1)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))
    ALL = list(range(X.shape[1]))
    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct, seed {a.seed}, {a.bags} GBM bags",
          flush=True)

    spans = []
    for qi in range(nq):
        w_ = np.nonzero(rowq == qi)[0]
        spans.append((int(w_[0]), int(w_[-1]) + 1) if len(w_) else (0, 0))
    C = lambda s: curves(qids, ngold, gis, spans, s)

    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for qi, q in enumerate(qids):
        w_ = np.nonzero(rowq == qi)[0]
        QF[qi, :4] = [1.0, math.log1p(len(w_)) / 5.0,
                      (cnt[w_].max() if len(w_) else 0) / n, (cnt[w_] >= 2).sum() / 20.0]
        QF[qi, 4 + tix.index(qtype[q])] = 1.0

    ARMS = ["count", "pattern", "RRF(K=10)", "nb-rank", "LEAN", "GBM(bagged)",
            "STACK", "LEAN + budget", "GBM(bagged) + budget", "STACK + budget"]
    per = {arm: np.zeros(nq) for arm in ARMS}
    diag = collections.defaultdict(list)

    rng = random.Random(a.seed)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    def budget(s, p_of, tr_q, te_q, arm):
        p = p_of(s)
        QG = np.zeros((nq, QF.shape[1] + 3))
        QG[:, :QF.shape[1]] = QF
        for qi in range(nq):
            w_ = np.nonzero(rowq == qi)[0]
            QG[qi, QF.shape[1]:] = [p[w_].sum() / 10.0,
                                    p[w_][cnt[w_] >= 2].sum() / 10.0,
                                    float(p[w_].max()) if len(w_) else 0.0]
        beta = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
        Gh = np.maximum(QG @ beta, 1.0)
        fa, _ = ev(adaptive_keep(p, rowq, nq, Gh), list(tr_q))
        for qi in te_q:
            per[arm][qi] = fa[qi]

    sig = lambda s: 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]

        cut = int(0.85 * len(tr_q))
        inner, held = tr_q[:cut], tr_q[cut:]
        im, hm = np.isin(rowq, inner), np.isin(rowq, held)

        # ---- base scorers, all fitted on `inner` only
        plut, cr = fit_pattern([p for p, m in zip(pats, im) if m], y[im])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        wl = fit_logistic(X[np.ix_(im, LEAN)], y[im], ridge=1.0)
        s_lean = X[:, LEAN] @ wl
        alr, olr, ilr, prior = fit_nb_rank(Z[im], B[im], y[im], models)
        s_nb = score_nb(Z, B, alr, olr, prior)
        fus = fusion_scores(RK, LL)
        s_rrf = fus["RRF(K=10)"]
        t1, _, b1 = lookup2d(cnt[im].astype(int), mb[im], y[im], NB_BUCKETS)
        s_2d = np.array([t1.get((int(c_), b_), b1) for c_, b_ in zip(cnt, mb)])
        t2, m2, b2 = lookup2d(code[im], mb[im], y[im], NB_BUCKETS)
        s_2dp = np.array([t2.get((c_, b_), m2.get(c_, b2)) for c_, b_ in zip(code, mb)])

        _, edges = bin_features(X[im][:, ALL])
        allc = np.zeros((len(X), len(ALL)), dtype=np.uint8)
        for j, e in enumerate(edges):
            allc[:, j] = np.searchsorted(e, X[:, ALL[j]], side="right")
        bagged = np.zeros(len(y)); ntr = []
        for bag in range(a.bags):
            g = GBM(n_trees=a.trees, lr=0.08, max_depth=5, subsample=0.8,
                    seed=100 * f + bag).fit(None, y[im], codes=allc[im],
                                            edges=edges, val=(allc[hm], y[hm]),
                                            patience=40)
            bagged += g.decision(allc); ntr.append(len(g.trees))
        s_gbm = bagged / a.bags

        # ---- meta model on the held-out quarter of train
        M = np.column_stack([np.ones(len(y)), cnt / n,
                             np.log(pat_p / (1 - pat_p)),
                             s_lean, s_gbm, s_nb,
                             s_rrf * 10.0, s_2d, s_2dp])
        wm = fit_logistic(M[hm], y[hm], ridge=1.0)
        s_stack = M @ wm

        for nm, s in (("ap_lean", s_lean), ("ap_gbm", s_gbm),
                      ("ap_stack", s_stack)):
            diag[nm].append(ap_score(s[~trm], y[~trm]))
        diag["ntrees"].append(int(statistics.mean(ntr)))

        for arm, s in (("count", cnt), ("pattern", pat_p), ("RRF(K=10)", s_rrf),
                       ("nb-rank", s_nb), ("LEAN", s_lean),
                       ("GBM(bagged)", s_gbm), ("STACK", s_stack)):
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        budget(s_lean, sig, tr_q, te_q, "LEAN + budget")
        budget(s_gbm, sig, tr_q, te_q, "GBM(bagged) + budget")
        budget(s_stack, sig, tr_q, te_q, "STACK + budget")
        print(f"    fold {f+1}/{FOLDS} done  (trees~{diag['ntrees'][-1]}, "
              f"AP lean {diag['ap_lean'][-1]:.4f} gbm {diag['ap_gbm'][-1]:.4f} "
              f"stack {diag['ap_stack'][-1]:.4f})", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(Z[:, j], list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])
    ref_c = list(per["count"]); ref_p = list(per["pattern"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"  average precision: LEAN {statistics.mean(diag['ap_lean']):.4f} | "
          f"GBM {statistics.mean(diag['ap_gbm']):.4f} | "
          f"STACK {statistics.mean(diag['ap_stack']):.4f}")
    print(f"\n  {'arm':<24}{'F1':>7}{'vs best single':>26}{'vs count':>26}"
          f"{'vs ceiling':>20}")
    print("  " + "-" * 103)

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    res = {}
    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c1 = f"{'(reference)':>26}" if arm == "count" else cell(v, ref_c)
        if arm == "pattern":
            c2 = f"{'(ceiling)':>20}"
        else:
            d, lo, hi = bootstrap(v, ref_p)
            c2 = f"{d:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<24}{res[arm]:7.2f}{cell(v, best_v):>26}{c1:>26}{c2:>20}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")

    json.dump(res, open(RESULTS / f"weighting_v8_seed{a.seed}.json", "w"), indent=2)
    print(f"  wrote results/weighting_v8_seed{a.seed}.json")


if __name__ == "__main__":
    main()
