"""
LEAVE-ONE-SOURCE-OUT: does the method transfer to question types it never trained on?

THE CONCERN (21 Aug meeting)
  "We are remembering patterns and that should artificially boost our
   performance, unless we have a separate training set and a separate testing
   set... it might be relying on the specific characteristics of the immediately
   present dataset."

  The existing protocol already splits train/test BY QUESTION, 5-fold, and fits
  every weight, coefficient, ridge and threshold on training folds only. So the
  literal version of the concern is already handled. But the sharper version is
  not: random folds draw train and test from the SAME question distribution, so
  a rule that exploits something peculiar to QAMPARI-as-a-whole would still look
  fine.

THE HARDER TEST
  QAMPARI ships five question SOURCES, and they differ structurally:

    wikidata_simple          single-hop lookups
    wikidata_intersection    answers satisfying two constraints at once
    wikidata_comp            compositional, two-hop
    wikitables_simple        drawn from tables rather than the graph
    wikitables_composition   compositional over tables

  Train on four, test on the fifth, five times. Nothing about the held-out
  question type is seen during fitting -- not the rank profile, not the
  verbosity distribution, not the gold-count regression. If the gain survives,
  "relying on characteristics of this dataset" is ruled out in the strong form,
  because each fold is effectively a different dataset.

  Reported against the SAME arms under random 5-fold, so the two protocols are
  directly comparable. A drop from random-fold to LOSO is the size of the
  distribution-shift penalty; no drop means the signals are general.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.loso
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern, GRID)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.gbm import GBM, bin_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

ARMS = ["count", "pattern", "LEAN", "LEAN + budget", "BLEND + budget"]


def run(qids, qtype, fold_of, X, y, rowq, pats, gis, RK, blocks, STACK,
        ev, ng_arr, cnt, C, QF, LEAN, ALL, nq, label):
    per = {a: np.zeros(nq) for a in ARMS}
    folds = sorted(set(fold_of.values()))
    for f in folds:
        tr_q = np.array([i for i in range(nq) if fold_of[qids[i]] != f])
        te_q = np.array([i for i in range(nq) if fold_of[qids[i]] == f])
        if not len(te_q) or not len(tr_q):
            continue
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]
        cut = int(0.85 * len(tr_q))
        inner, held = tr_q[:cut], tr_q[cut:]
        im, hm = np.isin(rowq, inner), np.isin(rowq, held)

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        wl = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
        s_lean = X[:, LEAN] @ wl

        _, edges = bin_features(X[im][:, ALL])
        allc = np.zeros((len(X), len(ALL)), dtype=np.uint8)
        for j, e in enumerate(edges):
            allc[:, j] = np.searchsorted(e, X[:, ALL[j]], side="right")
        gb = GBM(n_trees=500, lr=0.08, max_depth=5, subsample=0.8, seed=0).fit(
            None, y[im], codes=allc[im], edges=edges, val=(allc[hm], y[hm]),
            patience=40)
        s_gbm = gb.decision(allc)
        Zb = np.column_stack([np.ones(len(y)), s_lean, s_gbm])
        wb = fit_logistic(Zb[hm], y[hm], ridge=1.0)
        s_blend = Zb @ wb

        for arm, s in (("count", cnt), ("pattern", pat_p), ("LEAN", s_lean)):
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        for arm, s in (("LEAN + budget", s_lean), ("BLEND + budget", s_blend)):
            p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
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
        print(f"    {label} fold {f} done ({len(te_q)} test questions)", flush=True)
    return per


def main():
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))
    ALL = list(range(X.shape[1]))
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

    print("=" * 90)
    print("  LEAVE-ONE-SOURCE-OUT vs RANDOM 5-FOLD  (QAMPARI)")
    print("=" * 90)
    counts = collections.Counter(qtype[q] for q in qids)
    print(f"\n  question sources:")
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {t.replace('__dev',''):<26}{c:>5} questions")

    # ---- LOSO
    print(f"\n  leave-one-source-out")
    loso_fold = {q: qtype[q] for q in qids}
    per_loso = run(qids, qtype, loso_fold, X, y, rowq, pats, gis, RK, blocks,
                   STACK, ev, ng_arr, cnt, C, QF, LEAN, ALL, nq, "LOSO")

    # ---- random 5-fold, same arms, for comparison
    print(f"\n  random 5-fold (same arms, same code path)")
    rng = random.Random(0)
    order = qids[:]; rng.shuffle(order)
    rand_fold = {q: i % 5 for i, q in enumerate(order)}
    per_rand = run(qids, qtype, rand_fold, X, y, rowq, pats, gis, RK, blocks,
                   STACK, ev, ng_arr, cnt, C, QF, LEAN, ALL, nq, "random")

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(RK[:, j] >= 0, list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"\n  {'arm':<18}{'random 5-fold':>15}{'LOSO':>10}{'shift':>9}"
          f"{'LOSO vs count':>26}")
    print("  " + "-" * 78)
    rc = list(per_rand["count"]); lc = list(per_loso["count"])
    res = {}
    for arm in ARMS:
        r = 100 * per_rand[arm].mean(); l = 100 * per_loso[arm].mean()
        res[arm] = {"random": r, "loso": l}
        if arm == "count":
            cell = f"{'(reference)':>26}"
        else:
            d, lo, hi = bootstrap(list(per_loso[arm]), lc)
            cell = (f"{d:+6.2f} ({100*d/(100*statistics.mean(lc)):+5.1f}%)"
                    f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")
        print(f"  {arm:<18}{r:>15.2f}{l:>10.2f}{l-r:>+9.2f}{cell:>26}")

    print(f"\n  PER-SOURCE, held out entirely from training")
    print(f"  {'source':<26}{'n':>5}{'count':>9}{'BLEND+budget':>14}{'gain':>9}")
    print("  " + "-" * 63)
    for t in sorted(counts, key=lambda t: -counts[t]):
        idx = [i for i, q in enumerate(qids) if qtype[q] == t]
        c = 100 * per_loso["count"][idx].mean()
        b = 100 * per_loso["BLEND + budget"][idx].mean()
        print(f"  {t.replace('__dev',''):<26}{len(idx):>5}{c:>9.2f}{b:>14.2f}"
              f"{b-c:>+9.2f}")

    json.dump(res, open(RESULTS / "loso.json", "w"), indent=2)
    print("\n  wrote results/loso.json")


if __name__ == "__main__":
    main()
