"""
Nonlinear scoring. The ordering was the bottleneck, so replace the linear model.

WHERE v4 LEFT IT (QAMPARI, 777 q, 8 agents, 5-fold cross-fitted)
    count                    29.35
    pattern (ceiling)        29.60   +0.25  ns
    LEAN (logistic)          30.79   +1.44  *
    LEAN + budget            31.71   +2.36  *
    ORACLE selection         59.34  +29.99

  v4 ruled out the decision rule as the remaining lever: the logistic was already
  calibrated (3.55% predicted vs 3.55% actual, so isotonic changed nothing),
  within-question normalisation HURT (z-norm 28.41, pct-norm 26.79), and a
  constant per-question budget was worth nothing (top-K 29.41). What is left is
  the ORDERING, and the signals it has to combine are interactive in a way a
  linear model cannot express:

    rank matters more for a verbose agent than a terse one;
    a solo claim from a 3-item list is not a solo claim from a 2431-item list;
    a support count means different things depending on which agents and where.

  So: same features, same folds, same everything -- a gradient boosted model
  instead of a logistic. Then the budget rule on top of it.

ARMS
  count | pattern | LEAN(logistic) | GBM | GBM + budget
  + the two headroom diagnostics carried through from v4.

  Early stopping is on a held-out slice of the TRAINING fold. The test fold is
  not touched by fitting, binning, threshold selection or stopping.

A NOTE ON `oracle-k`, WHICH v3 REPORTED AT 39.90
  That picks the best prefix per question using that question's own labels, out
  of ~143 candidates. It is the maximum of many noisy values and is therefore
  heavily optimistic -- v4 showed the honest versions of it collect almost none
  of that: a learned k scored 25.64 and a constant k scored 29.41. Treat 39.90
  as an artefact of post-hoc selection, not as available headroom.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v5
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.gbm import GBM, bin_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS


def ap_score(p, y):
    """Average precision -- ranking quality, independent of any threshold."""
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(1.0, ys.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bench", default="qampari", choices=["qampari", "cwq", "webqsp", "quest"])
    ap.add_argument("--pool", default="", help="comma-separated agents (default: all)")
    ap.add_argument("--out", default="", help="write one JSON (means + per-question) here and nothing else")
    ap.add_argument("--no-gbm", action="store_true",
                    help="skip the boosted arms (not part of the method); used for subset sweeps")
    ap.add_argument("--ghat", default="log", choices=["linear", "log"],
                    help="answer-count estimate: ridge on G (paper) or on log G (heavy-tailed gold)")
    a = ap.parse_args()
    pool = [x for x in a.pool.split(",") if x] or None

    print("=" * 92)
    print("  WEIGHTING v5 -- gradient boosted scoring, then the per-question budget")
    print("=" * 92)
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich(pool, a.bench)
    print(f"  benchmark: {a.bench}; agents: {', '.join(models)}")
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))
    ALL = list(range(X.shape[1]))
    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct, {X.shape[1]} features")
    print(f"  GBM: {a.trees} trees max, depth {a.depth}, lr {a.lr}, "
          f"early stop on a held-out slice of train", flush=True)

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

    # "weighted" is reliability weighting (Naive Bayes over agents, as in the paper);
    # "* + budget" on count / weighted / pattern applies the SAME per-question
    # budget to scorers that read no generation behaviour (budget ablation).
    ARMS = ["count", "weighted", "pattern", "LEAN (logistic)",
            "count + budget", "weighted + budget", "pattern + budget", "LEAN + budget",
            "GBM", "GBM + budget", "BLEND", "BLEND + budget",
            "ORACLE selection"]
    if a.no_gbm:
        ARMS = [x for x in ARMS if not x.startswith(("GBM", "BLEND"))]
    mi_ = {m: j for j, m in enumerate(models)}
    Zs = np.zeros((len(pats), n))
    for i, ms in enumerate(pats):
        for m in ms:
            Zs[i, mi_[m]] = 1.0
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

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        wl = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
        s_lean = X[:, LEAN] @ wl

        if not a.no_gbm:
            # ---- GBM. Binning is fitted on train rows only.
            inner = tr_q[: int(0.85 * len(tr_q))]
            held = tr_q[int(0.85 * len(tr_q)):]
            im, hm = np.isin(rowq, inner), np.isin(rowq, held)
            codes_tr, edges = bin_features(X[trm][:, ALL])
            allc = np.zeros((len(X), len(ALL)), dtype=np.uint8)
            for j, e in enumerate(edges):
                allc[:, j] = np.searchsorted(e, X[:, ALL[j]], side="right")
            g = GBM(n_trees=a.trees, lr=a.lr, max_depth=a.depth,
                    subsample=0.8, seed=f).fit(
                        None, y[im], codes=allc[im], edges=edges,
                        val=(allc[hm], y[hm]))
            s_gbm = g.decision(allc)
            diag["ntrees"].append(len(g.trees))
            diag["ap_lean"].append(ap_score(s_lean[~trm], y[~trm]))
            diag["ap_gbm"].append(ap_score(s_gbm[~trm], y[~trm]))

            # BLEND: the two scorers disagree in useful ways -- the logistic is
            # better calibrated ACROSS questions, the GBM ranks better WITHIN them.
            # Combine them on the held-out slice, which the GBM did not train on.
            Zb = np.column_stack([np.ones(len(y)), s_lean, s_gbm])
            wb = fit_logistic(Zb[hm], y[hm], ridge=1.0)
            s_blend = Zb @ wb

            diag["ap_blend"].append(ap_score(s_blend[~trm], y[~trm]))
        # Reliability weighting as reported in the paper (weighting_schemes.py
        # "nb-full"): Naive Bayes log-likelihood ratio per agent, from train-fold
        # TPR_j = P(j asserts | correct) and FPR_j = P(j asserts | wrong), with
        # silence scored by log((1-TPR)/(1-FPR)).
        pos_, neg_ = y[trm].sum(), (1 - y[trm]).sum()
        tpr = ((Zs[trm] * y[trm, None]).sum(0) + 1.0) / (pos_ + 2.0)
        fpr = ((Zs[trm] * (1 - y[trm])[:, None]).sum(0) + 1.0) / (neg_ + 2.0)
        s_w = (np.log((pos_ + 1) / (neg_ + 1)) + Zs @ np.log(tpr / fpr)
               + (1 - Zs) @ np.log((1 - tpr) / (1 - fpr)))
        scores = {"count": cnt, "weighted": s_w, "pattern": pat_p, "LEAN (logistic)": s_lean}
        if not a.no_gbm:
            scores.update({"GBM": s_gbm, "BLEND": s_blend})
        for arm, s in scores.items():
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        # calibrated probabilities for the non-behavioural scorers, fitted on train
        cal = lambda sc: 1.0 / (1.0 + np.exp(-np.clip(
            np.column_stack([np.ones(len(sc)), sc]) @ fit_logistic(
                np.column_stack([np.ones(trm.sum()), sc[trm]]), y[trm], ridge=1.0), -30, 30)))
        probs = {"count + budget": cal(cnt / n), "weighted + budget": cal(s_w),
                 "pattern + budget": pat_p}
        budget_arms = [("count + budget", None), ("weighted + budget", None),
                       ("pattern + budget", None), ("LEAN + budget", s_lean)]
        if not a.no_gbm:
            budget_arms += [("GBM + budget", s_gbm), ("BLEND + budget", s_blend)]
        for arm, s in budget_arms:
            p = probs[arm] if s is None else 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
            QG = np.zeros((nq, QF.shape[1] + 3))
            QG[:, :QF.shape[1]] = QF
            for qi in range(nq):
                w_ = np.nonzero(rowq == qi)[0]
                QG[qi, QF.shape[1]:] = [p[w_].sum() / 10.0,
                                        p[w_][cnt[w_] >= 2].sum() / 10.0,
                                        float(p[w_].max()) if len(w_) else 0.0]
            if a.ghat == "log":   # gold counts are heavy-tailed (WebQSP mean ~76)
                beta = fit_gold_ridge(QG[tr_q], np.log(ng_arr[tr_q]))
                Gh = np.maximum(np.exp(QG @ beta), 1.0)
            else:
                beta = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
                Gh = np.maximum(QG @ beta, 1.0)
            fa, _ = ev(adaptive_keep(p, rowq, nq, Gh), list(tr_q))
            for qi in te_q:
                per[arm][qi] = fa[qi]

        fa, _ = ev(np.array([x is not None for x in gis]), list(tr_q))
        for qi in te_q:
            per["ORACLE selection"][qi] = fa[qi]
        print(f"    fold {f+1}/{FOLDS} done" + ("" if a.no_gbm else f"  ({len(g.trees)} trees, "
              f"AP {diag['ap_lean'][-1]:.4f} -> {diag['ap_gbm'][-1]:.4f})"), flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(RK[:, j] >= 0, list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])
    ref_c = list(per["count"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    if not a.no_gbm:
        print(f"  average precision on held-out claims: logistic "
              f"{statistics.mean(diag['ap_lean']):.4f} -> GBM "
              f"{statistics.mean(diag['ap_gbm']):.4f} -> blend "
              f"{statistics.mean(diag['ap_blend']):.4f}")
        print(f"  trees kept per fold: {diag['ntrees']}")
    print(f"\n  {'arm':<22}{'F1':>7}{'vs best single':>27}{'vs count':>27}")
    print("  " + "-" * 83)

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    res = {}
    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c1 = f"{'(reference)':>27}" if arm == "count" else cell(v, ref_c)
        print(f"  {arm:<22}{res[arm]:7.2f}{cell(v, best_v):>27}{c1:>27}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")

    if a.out:   # sweep mode: one self-contained file, nothing else touched
        json.dump({"bench": a.bench, "agents": models, "n_questions": nq, "f1": res,
                   "best_single": bm, "singles": {m: 100 * float(singles[m].mean()) for m in models},
                   "per_question": {arm: [float(x) for x in per[arm]] for arm in ARMS},
                   "best_single_per_question": [float(x) for x in best_v]}, open(a.out, "w"))
        return
    tag = ("" if a.bench == "qampari" and not pool else f"_{a.bench}" + (f"_{len(models)}agents" if pool else "")) \
        + ("_linearG" if a.ghat == "linear" else "") + ("_nogbm" if a.no_gbm else "")
    json.dump(res, open(RESULTS / f"weighting_v5{tag}_seed{a.seed}.json", "w"), indent=2)
    print(f"  wrote results/weighting_v5{tag}_seed{a.seed}.json")
    json.dump({"bench": a.bench, "agents": models, "best_single": bm,
               "singles": {m: 100 * float(singles[m].mean()) for m in models},
               "best_single_per_question": [float(x) for x in best_v]},
              open(RESULTS / f"weighting_v5{tag}_meta_seed{a.seed}.json", "w"), indent=1)

    # Per-question scores, kept so that power and equivalence analyses can be
    # run without refitting. `per[arm]` is one held-out F1 per question, in the
    # order of `qids`, which is what the paired bootstrap resamples.
    json.dump({"qids": list(qids),
               "per_question": {arm: [float(x) for x in per[arm]] for arm in ARMS}},
              open(RESULTS / f"weighting_v5{tag}_perq_seed{a.seed}.json", "w"))
    print(f"  wrote results/weighting_v5{tag}_perq_seed{a.seed}.json")


if __name__ == "__main__":
    main()
