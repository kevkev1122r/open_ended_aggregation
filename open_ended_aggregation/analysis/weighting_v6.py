"""
Three families we had not touched: IR rank fusion, pairwise co-assertion, 2D lookups.

WHY THESE THREE

  RANK FUSION.  Reciprocal Rank Fusion, Borda, CombSUM and CombMNZ are the
  standard way the IR literature merges ranked lists from several systems. That
  is exactly our problem -- eight agents each return an ordered list -- and we
  have not tried any of them. They are also the right BASELINES: a reviewer will
  ask why a bespoke learned scorer beats RRF, and right now we could not say.
  RRF in particular is parameter-free apart from K and is famously hard to beat.

  PAIRWISE CO-ASSERTION.  weighting_v2 tested error correlation through the
  hand-written form score = SUM w_m z_m - lambda SUM rho_ij z_i z_j, and lambda
  came out 0 in every fold. That tests one functional form, not the idea. Giving
  the model all C(8,2)=28 co-assertion indicators lets it learn which specific
  pairs are worth discounting, rather than assuming a single global lambda.

  2D LOOKUPS.  `pattern` is P(correct | who asserted) and is the ceiling for
  who-said-what rules. `rank` beat it by conditioning on something else. The
  obvious next object is the joint: P(correct | who asserted, where they put
  it). Nonparametric, no functional form assumed, shrunk toward the marginal.

  RANK AGREEMENT.  Do the agents agree on ORDER, not just on membership? Spread
  of ranks across supporters, and the range. Two agents both ranking a claim
  first is a different event from one ranking it first and one ranking it 40th.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v6
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import (
    bucket, NB_BUCKETS, fit_gold_ridge, adaptive_keep)
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS


def rank_tensors(models, qids, cand, nlist):
    """rank matrix (-1 = silent) and the asserting agent's list length."""
    n = len(models)
    mi = {m: i for i, m in enumerate(models)}
    RK, LL, rq = [], [], []
    for qi, q in enumerate(qids):
        for ms, gi, rk, key in cand[q]:
            r = np.full(n, -1.0); l = np.zeros(n)
            for m in ms:
                r[mi[m]] = rk[m]; l[mi[m]] = nlist[q][m]
            RK.append(r); LL.append(l); rq.append(qi)
    return np.array(RK), np.array(LL), np.array(rq)


def fusion_scores(RK, LL, w=None):
    """Classic list-fusion rules. Z = asserted mask."""
    Z = RK >= 0
    L = np.maximum(LL, 1.0)
    out = {}
    for K in (10.0, 60.0):
        out[f"RRF(K={int(K)})"] = np.where(Z, 1.0 / (K + RK), 0.0).sum(1)
    # Borda: credit for how far from the BOTTOM of that agent's list
    out["Borda"] = np.where(Z, (L - RK) / L, 0.0).sum(1)
    combsum = np.where(Z, 1.0 - RK / L, 0.0).sum(1)
    out["CombSUM"] = combsum
    out["CombMNZ"] = combsum * Z.sum(1)
    if w is not None:
        out["wRRF"] = (np.where(Z, 1.0 / (60.0 + RK), 0.0) * w).sum(1)
        out["wBorda"] = (np.where(Z, (L - RK) / L, 0.0) * w).sum(1)
    return out


def lookup2d(keys, buckets, y, nb, alpha=20.0):
    """P(correct | key, rank bucket), shrunk toward the key's marginal rate."""
    marg = collections.defaultdict(lambda: [0.0, 0.0])
    joint = collections.defaultdict(lambda: [0.0, 0.0])
    for k, b, t in zip(keys, buckets, y):
        marg[k][0] += t; marg[k][1] += 1
        joint[(k, b)][0] += t; joint[(k, b)][1] += 1
    mr = {k: (v[0] + 1) / (v[1] + 2) for k, v in marg.items()}
    base = (sum(v[0] for v in marg.values()) + 1) / (sum(v[1] for v in marg.values()) + 2)
    tbl = {kb: (v[0] + alpha * mr.get(kb[0], base)) / (v[1] + alpha)
           for kb, v in joint.items()}
    return tbl, mr, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    print("=" * 96)
    print("  WEIGHTING v6 -- IR rank fusion, pairwise co-assertion, 2D lookups")
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
    minr = np.where(Z, RK, np.inf).min(1)
    mb = bucket(np.where(np.isfinite(minr), minr, 0).astype(int))
    code = (Z * (1 << np.arange(n))).sum(1)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))

    # ---- extra feature blocks
    PAIRS = list(itertools.combinations(range(n), 2))
    PW = np.zeros((len(y), len(PAIRS)))
    for c_, (i_, j_) in enumerate(PAIRS):
        PW[:, c_] = (Z[:, i_] & Z[:, j_]).astype(float)
    rk_valid = np.where(Z, RK, np.nan)
    with np.errstate(invalid="ignore"):
        rstd = np.nan_to_num(np.nanstd(rk_valid, axis=1))
        rrng = np.nan_to_num(np.nanmax(rk_valid, axis=1) - np.nanmin(rk_valid, axis=1))
    AG = np.column_stack([np.log1p(rstd) / 5.0, np.log1p(rrng) / 5.0,
                          np.log1p(rstd) / 5.0 * cnt / n])
    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct")
    print(f"  extra blocks: {len(PAIRS)} pairwise co-assertion, "
          f"{AG.shape[1]} rank-agreement", flush=True)

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

    FUSION = ["RRF(K=10)", "RRF(K=60)", "Borda", "CombSUM", "CombMNZ",
              "wRRF", "wBorda"]
    ARMS = (["count", "pattern"] + FUSION +
            ["2D: count x rank", "2D: pattern x rank",
             "LEAN", "LEAN + pairwise", "LEAN + agreement", "LEAN + both",
             "LEAN + both + budget", "RRF + budget"])
    per = {arm: np.zeros(nq) for arm in ARMS}

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

        # solo precision per agent, for the weighted fusion variants
        sn = np.zeros(n); sc = np.zeros(n)
        solo = (cnt == 1) & trm
        for j in range(n):
            m_ = solo & Z[:, j]
            sn[j] = m_.sum(); sc[j] = y[m_].sum()
        wsolo = (sc + 1) / (sn + 2)

        scores = {"count": cnt, "pattern": pat_p}
        scores.update(fusion_scores(RK, LL, wsolo))

        t1, _, b1 = lookup2d(cnt[trm].astype(int), mb[trm], y[trm], NB_BUCKETS)
        scores["2D: count x rank"] = np.array(
            [t1.get((int(c_), b_), b1) for c_, b_ in zip(cnt, mb)])
        t2, m2, b2 = lookup2d(code[trm], mb[trm], y[trm], NB_BUCKETS)
        scores["2D: pattern x rank"] = np.array(
            [t2.get((c_, b_), m2.get(c_, b2)) for c_, b_ in zip(code, mb)])

        variants = {
            "LEAN": X[:, LEAN],
            "LEAN + pairwise": np.column_stack([X[:, LEAN], PW]),
            "LEAN + agreement": np.column_stack([X[:, LEAN], AG]),
            "LEAN + both": np.column_stack([X[:, LEAN], PW, AG]),
        }
        fitted = {}
        for arm, M in variants.items():
            w_ = fit_logistic(M[trm], y[trm], ridge=1.0)
            fitted[arm] = M @ w_
            scores[arm] = fitted[arm]

        for arm, s in scores.items():
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        for arm, s in (("LEAN + both + budget", fitted["LEAN + both"]),
                       ("RRF + budget", scores["RRF(K=60)"])):
            p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30))) if "LEAN" in arm else \
                np.clip(s / max(1e-9, np.quantile(s[trm], 0.999)), 0, 1)
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
        print(f"    fold {f+1}/{FOLDS} done", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(Z[:, j], list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best_v = list(singles[bm])
    ref_c = list(per["count"])
    ref_p = list(per["pattern"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"\n  {'arm':<24}{'F1':>7}{'vs best single':>26}{'vs count':>26}{'vs ceiling':>20}")
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

    json.dump(res, open(RESULTS / "weighting_v6.json", "w"), indent=2)
    print("  wrote results/weighting_v6.json")


if __name__ == "__main__":
    main()
