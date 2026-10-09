"""
Does WHERE a claim sits in an agent's answer carry real information?

rank_effect.py (QAMPARI only) found the marginal gradient is steep (29% correct
at position 1, 0.2% at 71+) but mostly support and length: among solo
assertions the within-length-band Spearman is only -0.05 to -0.10, and on 1-5
item lists it reverses. This file asks the three things that decide whether the
method should lean on position at all.

T1  DOES THE METHOD USE IT?  Zero-label 4|4 pool transfer on QAMPARI (the
    headline setting, pool_transfer.py mode `self`), with feature groups
    ablated: rank, supporters' verbosity, silent agents' omission.

T2  IS IT THERE AFTER CONTROLS?  Per (claim, asserting agent) row, a logistic
    regression  correct ~ support by others + log list length + relative rank
    (+ absolute log rank), on QAMPARI, CWQ and WebQSP. Question-level bootstrap
    CI on the rank coefficients.

T3  IS IT A GOLD ARTEFACT?  Famous entities are generated first AND are the
    ones an incomplete gold list is most likely to contain, so position could be
    predicting "in the gold list" rather than "true". WebQSP's gold is near-
    exhaustive (Freebase, mean 76 answers), so if the gradient is a gold-
    coverage artefact it should shrink there. Same model, compared across
    datasets, alongside how complete each gold list is.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.position_signal
"""
import json, random, itertools, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.pool_transfer import (
    load, view, AZURE, GPT, SEED, boot, inv_features, self_norm, qfeat, sig, Fitted)
from open_ended_aggregation.analysis.rank_crossover import Scorer, tensors
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.paths import RESULTS

# column groups of inv_features(v, med=self_norm(v))
GROUPS = {
    "rank":      [3, 4, 5, 6, 12, 13],
    "verbosity": [7, 14],
    "omission":  [8, 9, 15, 16],
}


def fit_apply(vS, trm, vT, keep_cols):
    FS = inv_features(vS, med=self_norm(vS))[:, keep_cols]
    FT = inv_features(vT, med=self_norm(vT))[:, keep_cols]
    m = trm[vS["rq"]]
    w = fit_logistic(FS[m], vS["y"][m])
    pS, pT = sig(FS @ w), sig(FT @ w)
    qi = np.nonzero(trm)[0]
    beta = fit_gold_ridge(qfeat(vS, pS)[qi], np.log(vS["ng"][qi]))
    G = np.exp(qfeat(vT, pT) @ beta)
    sc = Scorer(vT["rq"], vT["rg"], vT["ng"], vT["nq"])
    return sc.f1_of_mask(adaptive_keep(pT, vT["rq"], vT["nq"], G))


def t1():
    models, qids, cand, ngold, nlist = load("qampari", AZURE + GPT)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]
    ncol = inv_features(view(T, [0, 1, 2, 3]), med=np.zeros(4)).shape[1]
    arms = {"full": [],
            "- rank": GROUPS["rank"],
            "- verbosity": GROUPS["verbosity"],
            "- omission": GROUPS["omission"],
            "- rank - verbosity - omission": sum(GROUPS.values(), [])}
    res = {k: [] for k in arms}
    for A in itertools.combinations(range(8), 4):
        B = [j for j in range(8) if j not in A]
        vA, vB = view(T, list(A)), view(T, B)
        cnt0 = np.zeros(nq)
        per = {k: np.zeros(nq) for k in arms}
        for tr, te in folds:
            trm = np.zeros(nq, bool); trm[tr] = True
            for k, drop in arms.items():
                keep = [c for c in range(ncol) if c not in drop]
                per[k][te] = fit_apply(vA, trm, vB, keep)[te]
            f = Fitted(vA, trm, "none")
            cnt0[te] = f.apply(vB)[0]["cnt-thr"][te]
        for k in arms:
            d, lo, hi = boot(per[k], per["full"])
            d0, lo0, hi0 = boot(per[k], cnt0)
            res[k].append(dict(f1=100 * per[k].mean(), vs_full=d, full_lo=lo, full_hi=hi,
                               vs_cnt=d0, cnt_lo=lo0))
    return res


def rows_for(name, pool):
    """One row per (claim, asserting agent)."""
    models, qids, cand, ngold, nlist = load(name, pool)
    X, y, g = [], [], []
    for qi, q in enumerate(qids):
        for ms, gi, rk in cand[q]:
            for m in ms:
                L = nlist[q][m]
                X.append([len(ms) - 1, np.log(L), rk[m] / L, np.log1p(rk[m])])
                y.append(0.0 if gi is None else 1.0)
                g.append(qi)
    X, y, g = np.array(X), np.array(y), np.array(g)
    gold_cov = None
    return X, y, g, len(qids)


def t2_t3():
    out = {}
    for name, pool in (("qampari", AZURE + GPT), ("cwq", AZURE), ("webqsp", AZURE)):
        X, y, g, nq = rows_for(name, pool)
        # standardise the two rank terms so coefficients are comparable across sets
        def design(X):
            sup, ll, rr, lr = X.T
            return np.column_stack([np.ones(len(X)), sup, ll, rr, lr])
        D = design(X)
        w = fit_logistic(D, y, ridge=1e-3)
        rng = np.random.default_rng(0)
        byq = [np.nonzero(g == q)[0] for q in range(nq)]
        reps = []
        for _ in range(200):
            idx = np.concatenate([byq[i] for i in rng.integers(0, nq, nq)])
            reps.append(fit_logistic(D[idx], y[idx], ridge=1e-3))
        reps = np.array(reps)
        lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
        # effect size: predicted P(correct) at 1st vs last quintile of the list,
        # for a solo claim in a 20-item list (the typical mid-length case)
        def p_at(rel, sup=0, L=20):
            x = np.array([1, sup, np.log(L), rel, np.log1p(rel * L)])
            return 1 / (1 + np.exp(-(x @ w)))
        # solo rows, within 6-100 item lists: precision by relative-rank quintile
        solo = (X[:, 0] == 0) & (np.exp(X[:, 1]) >= 6) & (np.exp(X[:, 1]) <= 100)
        quint = np.minimum((X[:, 2] * 5).astype(int), 4)
        tab = [float(100 * y[solo & (quint == k)].mean()) if (solo & (quint == k)).sum() > 50
               else None for k in range(5)]
        multi = (X[:, 0] >= 1) & (np.exp(X[:, 1]) >= 6) & (np.exp(X[:, 1]) <= 100)
        tabm = [float(100 * y[multi & (quint == k)].mean()) if (multi & (quint == k)).sum() > 50
                else None for k in range(5)]
        out[name] = dict(n_rows=int(len(y)), base=float(100 * y.mean()),
                         coef=dict(zip(["const", "support", "loglen", "relrank", "logrank"],
                                       map(float, w))),
                         ci={k: [float(a), float(b)] for k, a, b in
                             zip(["const", "support", "loglen", "relrank", "logrank"], lo, hi)},
                         p_first=float(100 * p_at(0.0)), p_last=float(100 * p_at(0.9)),
                         solo_by_quintile=tab, multi_by_quintile=tabm)
    return out


def main():
    res1 = t1()
    n = len(res1["full"])
    print(f"\n  T1  DOES THE METHOD USE IT?  zero-label QAMPARI 4|4 transfer, {n} splits")
    for k, R in res1.items():
        m = lambda x: statistics.mean(r[x] for r in R)
        worse = sum(1 for r in R if r["full_hi"] < 0)
        beat = sum(1 for r in R if r["cnt_lo"] > 0)
        print(f"    {k:<32} F1 {m('f1'):6.2f}   vs full {m('vs_full'):+5.2f}"
              f" (sig worse {worse}/{n})   vs zero-label count {m('vs_cnt'):+5.2f}"
              f" (sig wins {beat}/{n})")

    res2 = t2_t3()
    print("\n  T2/T3  correct ~ support + log length + relative rank + log rank,"
          " one row per (claim, asserting agent)")
    for name, r in res2.items():
        c, ci = r["coef"], r["ci"]
        print(f"\n    {name}: {r['n_rows']:,} rows, base rate {r['base']:.1f}%")
        for k in ("support", "loglen", "relrank", "logrank"):
            print(f"      {k:<8} {c[k]:+.3f}  [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")
        print(f"      model: solo claim, 20-item list -> P(correct) {r['p_first']:.1f}% at the "
              f"top vs {r['p_last']:.1f}% at 90% depth")
        fmt = lambda t: "  ".join("   -  " if v is None else f"{v:5.1f}%" for v in t)
        print(f"      raw, 6-100 item lists, by depth quintile  solo:  {fmt(r['solo_by_quintile'])}")
        print(f"                                                shared: {fmt(r['multi_by_quintile'])}")
    json.dump({"t1": res1, "t2": res2}, open(RESULTS / "position_signal.json", "w"),
              indent=1, default=float)
    print("\n  wrote results/position_signal.json")


if __name__ == "__main__":
    main()
