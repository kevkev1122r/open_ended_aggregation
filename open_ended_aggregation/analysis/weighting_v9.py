"""
The budget surrogate is wrong in a specific, fixable way: recall saturates.

THE BUG IN THE OBJECTIVE
  Every budget arm so far maximises

      2 * S_k / (k + G),        S_k = sum of the top-k predicted probabilities

  which treats S_k as the expected number of GOLD ANSWERS COVERED. It is not.
  It is the expected number of CORRECT CLAIMS kept, and several kept claims can
  be the same gold answer under different spellings -- QAMPARI's agents produce
  exactly that, which is why alias merging was worth testing at all. Recall
  counts distinct gold; precision counts every claim. So the true numerator
  saturates in S_k while the surrogate stays linear, and the rule systematically
  keeps too many claims.

THE FIX, FITTED RATHER THAN ASSUMED
  On the training questions, measure the actual distinct-gold coverage H at
  every prefix and regress

      log H  =  log alpha + beta * log S_k

  beta < 1 is the saturation. Then maximise 2*H_hat(S_k)/(k*c + G_hat), with a
  scalar c on the precision term also fitted on train, since the F1 optimum is
  sensitive to how hard length is punished.

ARMS
  budget                 the current rule, 2*S_k/(k+G)
  budget + c             one fitted scalar on the precision term
  budget + saturation    fitted beta, c = 1
  budget + both          both
  oracle-k               headroom on this ordering (selection-biased, see v5)

  Applied to the two best scorers, LEAN and BLEND, so the comparison is against
  the current champion rather than against a weak base.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v9
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.gbm import GBM, bin_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS


def prefix_tables(p, rowq, nq):
    """Per question, indices sorted by descending p and the cumulative sum."""
    o = np.lexsort((-p, rowq))
    qs = rowq[o]
    start = np.searchsorted(qs, np.arange(nq), side="left")
    end = np.searchsorted(qs, np.arange(nq), side="right")
    return o, start, end


def fit_saturation(p, rowq, nq, gis, train_q):
    """log H = log alpha + beta log S over training prefixes."""
    o, start, end = prefix_tables(p, rowq, nq)
    xs, ys = [], []
    for qi in train_q:
        a, b = start[qi], end[qi]
        if b <= a:
            continue
        idx = o[a:b]
        cs = np.cumsum(p[idx])
        seen, H = set(), []
        for i in idx:
            g = gis[i]
            if g is not None:
                seen.add(g)
            H.append(len(seen))
        H = np.array(H, dtype=float)
        m = (H > 0) & (cs > 0)
        if m.any():
            xs.append(np.log(cs[m])); ys.append(np.log(H[m]))
    if not xs:
        return 1.0, 1.0
    X_ = np.concatenate(xs); Y_ = np.concatenate(ys)
    A = np.column_stack([np.ones(len(X_)), X_])
    coef = np.linalg.lstsq(A, Y_, rcond=None)[0]
    return float(math.exp(coef[0])), float(coef[1])


def budget_mask(p, rowq, nq, G, c=1.0, alpha=1.0, beta=1.0):
    o, start, end = prefix_tables(p, rowq, nq)
    mask = np.zeros(len(p), dtype=bool)
    for qi in range(nq):
        a, b = start[qi], end[qi]
        if b <= a:
            continue
        idx = o[a:b]
        cs = np.cumsum(p[idx])
        k = np.arange(1, b - a + 1)
        H = alpha * np.power(np.maximum(cs, 1e-12), beta) if beta != 1.0 else cs
        val = 2 * H / (k * c + max(G[qi], 1e-6))
        j = int(np.argmax(val))
        if val[j] > 0:
            mask[idx[:j + 1]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    print("=" * 96)
    print("  WEIGHTING v9 -- fixing the budget objective (saturation + precision scale)")
    print("=" * 96)
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
    print(f"\n  {nq} questions, {len(y):,} claims, seed {a.seed}", flush=True)

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

    ARMS = ["count", "BLEND",
            "BLEND + budget", "BLEND + budget+c", "BLEND + budget+sat",
            "BLEND + budget+both", "BLEND + oracle-k"]
    per = {arm: np.zeros(nq) for arm in ARMS}
    diag = collections.defaultdict(list)

    rng = random.Random(a.seed)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
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
        gb = GBM(n_trees=500, lr=0.08, max_depth=5, subsample=0.8, seed=f).fit(
            None, y[im], codes=allc[im], edges=edges, val=(allc[hm], y[hm]),
            patience=40)
        s_gbm = gb.decision(allc)
        Zb = np.column_stack([np.ones(len(y)), s_lean, s_gbm])
        wb = fit_logistic(Zb[hm], y[hm], ridge=1.0)
        s_blend = Zb @ wb
        p = 1.0 / (1.0 + np.exp(-np.clip(s_blend, -30, 30)))

        for arm, s in (("count", cnt), ("BLEND", s_blend)):
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        QG = np.zeros((nq, QF.shape[1] + 3))
        QG[:, :QF.shape[1]] = QF
        for qi in range(nq):
            w_ = np.nonzero(rowq == qi)[0]
            QG[qi, QF.shape[1]:] = [p[w_].sum() / 10.0,
                                    p[w_][cnt[w_] >= 2].sum() / 10.0,
                                    float(p[w_].max()) if len(w_) else 0.0]
        beta_g = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
        Gh = np.maximum(QG @ beta_g, 1.0)
        alpha, beta = fit_saturation(p, rowq, nq, gis, tr_q)
        diag["beta"].append(beta); diag["alpha"].append(alpha)

        def pick(cgrid, bgrid):
            best = None
            for c in cgrid:
                for b_ in bgrid:
                    _, v = ev(budget_mask(p, rowq, nq, Gh, c, alpha, b_), list(tr_q))
                    if best is None or v > best[0]:
                        best = (v, c, b_)
            return best[1], best[2]

        combos = {
            "BLEND + budget": (1.0, 1.0),
            "BLEND + budget+c": pick([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], [1.0]),
            "BLEND + budget+sat": pick([1.0], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "BLEND + budget+both": pick([0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                                        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        }
        for arm, (c, b_) in combos.items():
            fa, _ = ev(budget_mask(p, rowq, nq, Gh, c, alpha, b_), list(tr_q))
            for qi in te_q:
                per[arm][qi] = fa[qi]
            if arm != "BLEND + budget":
                diag[arm].append((round(c, 2), round(b_, 2)))

        S, F = C(s_blend)
        for qi in te_q:
            per["BLEND + oracle-k"][qi] = float(F[qi].max())
        print(f"    fold {f+1}/{FOLDS} done  (beta={beta:.3f}, "
              f"c/beta picked {combos['BLEND + budget+both']})", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(RK[:, j] >= 0, list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])
    ref_c = list(per["count"])
    ref_b = list(per["BLEND + budget"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"  fitted saturation exponent beta = "
          f"{statistics.mean(diag['beta']):.3f} (1.0 = no saturation)")
    for k in ("BLEND + budget+c", "BLEND + budget+sat", "BLEND + budget+both"):
        print(f"  {k}: {diag[k]}")
    print(f"\n  {'arm':<24}{'F1':>7}{'vs best single':>26}{'vs count':>26}"
          f"{'vs plain budget':>24}")
    print("  " + "-" * 107)

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    res = {}
    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c1 = f"{'(reference)':>26}" if arm == "count" else cell(v, ref_c)
        if arm == "BLEND + budget":
            c2 = f"{'(reference)':>24}"
        else:
            d, lo, hi = bootstrap(v, ref_b)
            c2 = f"{d:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<24}{res[arm]:7.2f}{cell(v, best_v):>26}{c1:>26}{c2:>24}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")

    json.dump(res, open(RESULTS / f"weighting_v9_seed{a.seed}.json", "w"), indent=2)
    print(f"  wrote results/weighting_v9_seed{a.seed}.json")


if __name__ == "__main__":
    main()
