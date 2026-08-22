"""
The decision rule, not the score. This is where the remaining gain is.

THE DIAGNOSTIC THAT REDIRECTED THIS
  analysis/weighting_v3.py, QAMPARI, 5-fold cross-fitted:

    count                     29.35
    pattern (ceiling)         29.60   +0.25  ns
    v3 LEAN                   30.98   +1.63  *
    LEAN + budget             31.75   +2.40  *
    LEAN + budget (oracle G)  32.30   +2.94
    LEAN + ORACLE-k           39.89  +10.54          <-- same ordering!
    ORACLE selection          59.34  +29.99

  `LEAN + oracle-k` keeps the top-k of the SAME ranking the model already
  produces, with k chosen per question by an oracle. It scores 39.89. So the
  ordering is not the bottleneck -- choosing how many claims to keep is. The
  plug-in rule collects 2.40 of that 10.54, and giving it the true gold count
  only takes it to 2.94, which means the RULE is wrong, not its G input.

WHY THE PLUG-IN RULE UNDERPERFORMS
  k* = argmax_k 2*(sum_{i<=k} p_i)/(k + G) is the F1 optimum only if the p_i are
  calibrated. A logistic fitted on 3.5%-positive data with L2 shrinkage is not:
  it is systematically compressed toward the base rate, which flattens the
  objective and makes the argmax land almost anywhere. Two fixes, both cheap.

  CALIBRATION   isotonic (PAVA) map from raw p to empirical correctness,
                fitted on the training fold, then plugged into the same rule.

  NORMALISATION Instead of choosing k, make the SCORE comparable across
                questions and keep one global threshold. z-score, max-offset and
                within-question percentile each turn a global threshold into a
                per-question rule -- and unlike a budget they need no G at all.

  Plus `top-K`, the crudest possible budget (same k everywhere, cross-fitted),
  as the control that says how much of the gain is adaptivity per se rather than
  simply keeping fewer things.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v4
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import (
    fit_gold_ridge, adaptive_keep)
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

NBINS = 250


def isotonic_fit(p, y, nbins=NBINS):
    """Quantile-bin, then pool adjacent violators. Returns (x, v) for interp."""
    q = np.quantile(p, np.linspace(0, 1, nbins + 1))
    q = np.unique(q)
    if len(q) < 3:
        return np.array([0.0, 1.0]), np.array([y.mean(), y.mean()])
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    cnt = np.bincount(b, minlength=len(q) - 1).astype(float)
    sm = np.bincount(b, weights=y, minlength=len(q) - 1)
    keep = cnt > 0
    xs = ((q[:-1] + q[1:]) / 2)[keep]
    ys = (sm[keep] / cnt[keep])
    ws = cnt[keep]
    vv, ww, nn = [], [], []
    for i in range(len(ys)):
        cv, cw, cn = ys[i], ws[i], 1
        while vv and vv[-1] > cv:
            pv, pw, pn = vv.pop(), ww.pop(), nn.pop()
            cv = (pv * pw + cv * cw) / (pw + cw); cw = pw + cw; cn += pn
        vv.append(cv); ww.append(cw); nn.append(cn)
    out = np.empty(len(ys)); pos = 0
    for v_, n_ in zip(vv, nn):
        out[pos:pos + n_] = v_; pos += n_
    return xs, out


def isotonic_apply(xs, vs, p):
    return np.interp(p, xs, vs, left=vs[0], right=vs[-1])


def qnorm(s, rowq, nq, mode):
    """Make scores comparable across questions."""
    out = np.zeros_like(s)
    for qi in range(nq):
        w = np.nonzero(rowq == qi)[0]
        if not len(w):
            continue
        v = s[w]
        if mode == "z":
            sd = v.std()
            out[w] = (v - v.mean()) / (sd if sd > 1e-9 else 1.0)
        elif mode == "max":
            out[w] = v - v.max()
        elif mode == "pct":
            r = np.argsort(np.argsort(-v))
            out[w] = 1.0 - r / max(1, len(v))
    return out


def topk_mask(s, rowq, nq, k):
    m = np.zeros(len(s), dtype=bool)
    for qi in range(nq):
        w = np.nonzero(rowq == qi)[0]
        if not len(w):
            continue
        m[w[np.argsort(-s[w])[:k]]] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="")
    a = ap.parse_args()
    pool = [x for x in a.pool.split(",") if x] or None

    print("=" * 104)
    print("  WEIGHTING v4 -- fixing the DECISION RULE (calibration, "
          "within-question normalisation, budgets)")
    print("=" * 104)
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich(pool)
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))
    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct")

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

    ARMS = ["count", "pattern", "LEAN",
            "LEAN z-norm", "LEAN max-norm", "LEAN pct-norm",
            "LEAN top-K", "LEAN + budget", "LEAN + budget(calib)",
            "LEAN pct + budget(calib)",
            "LEAN + oracle-k", "ORACLE selection"]
    per = {a: np.zeros(nq) for a in ARMS}
    diag = collections.defaultdict(list)

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        w = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
        s_lean = X[:, LEAN] @ w
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(s_lean, -30, 30)))
        xs, vs = isotonic_fit(p_raw[trm], y[trm])
        p_cal = isotonic_apply(xs, vs, p_raw)
        diag["calib_raw"].append(float(p_raw[trm].mean()))
        diag["calib_cal"].append(float(p_cal[trm].mean()))
        diag["actual"].append(float(y[trm].mean()))

        scores = {
            "count": cnt, "pattern": pat_p, "LEAN": s_lean,
            "LEAN z-norm": qnorm(s_lean, rowq, nq, "z"),
            "LEAN max-norm": qnorm(s_lean, rowq, nq, "max"),
            "LEAN pct-norm": qnorm(s_lean, rowq, nq, "pct"),
        }
        for arm, s in scores.items():
            S, F = C(s)
            g = np.unique(s[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), g)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        # crudest budget: the same k on every question
        best = None
        for k in range(1, 41):
            _, v = ev(topk_mask(s_lean, rowq, nq, k), list(tr_q))
            if best is None or v > best[0]:
                best = (v, k)
        fa, _ = ev(topk_mask(s_lean, rowq, nq, best[1]), list(tr_q))
        for qi in te_q:
            per["LEAN top-K"][qi] = fa[qi]
        diag["topK"].append(best[1])

        QG = np.zeros((nq, QF.shape[1] + 3))
        QG[:, :QF.shape[1]] = QF
        for qi in range(nq):
            w_ = np.nonzero(rowq == qi)[0]
            QG[qi, QF.shape[1]:] = [p_cal[w_].sum() / 10.0,
                                    p_cal[w_][cnt[w_] >= 2].sum() / 10.0,
                                    float(p_cal[w_].max()) if len(w_) else 0.0]
        beta = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
        Ghat = np.maximum(QG @ beta, 1.0)

        for arm, p in (("LEAN + budget", p_raw), ("LEAN + budget(calib)", p_cal)):
            fa, _ = ev(adaptive_keep(p, rowq, nq, Ghat), list(tr_q))
            for qi in te_q:
                per[arm][qi] = fa[qi]

        # budget on the percentile-normalised ordering (same ordering, but the
        # calibrated probabilities decide the cut)
        fa, _ = ev(adaptive_keep(p_cal, rowq, nq, Ghat), list(tr_q))
        S, F = C(scores["LEAN pct-norm"])
        gg = np.unique(scores["LEAN pct-norm"][trm])
        if len(gg) > GRID:
            gg = np.quantile(gg, np.linspace(0, 1, GRID))
        th, _ = sweep(S, F, list(tr_q), gg)
        keep_pct = scores["LEAN pct-norm"] >= th - 1e-12
        keep_bud = adaptive_keep(p_cal, rowq, nq, Ghat)
        fa2, _ = ev(keep_pct & keep_bud, list(tr_q))
        for qi in te_q:
            per["LEAN pct + budget(calib)"][qi] = fa2[qi]

        S, F = C(s_lean)
        for qi in te_q:
            per["LEAN + oracle-k"][qi] = float(F[qi].max())
        fa, _ = ev(np.array([g is not None for g in gis]), list(tr_q))
        for qi in te_q:
            per["ORACLE selection"][qi] = fa[qi]
        print(f"    fold {f+1}/{FOLDS} done", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(RK[:, j] >= 0, list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])
    ref_c = list(per["count"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"  calibration on train: predicted {100*statistics.mean(diag['calib_raw']):.2f}% "
          f"raw / {100*statistics.mean(diag['calib_cal']):.2f}% calibrated "
          f"vs {100*statistics.mean(diag['actual']):.2f}% actual")
    print(f"  top-K picked per fold: {diag['topK']}")
    print(f"\n  {'arm':<26}{'F1':>7}{'vs best single':>26}{'vs count':>26}")
    print("  " + "-" * 85)

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    res = {}
    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c1 = f"{'(reference)':>26}" if arm == "count" else cell(v, ref_c)
        print(f"  {arm:<26}{res[arm]:7.2f}{cell(v, best_v):>26}{c1:>26}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")
    print("  oracle-k and ORACLE selection are headroom diagnostics, not methods")

    json.dump(res, open(RESULTS / "weighting_v4.json", "w"), indent=2)
    print("  wrote results/weighting_v4.json")


if __name__ == "__main__":
    main()
