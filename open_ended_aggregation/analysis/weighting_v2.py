"""
Second round of weighting mechanisms. Cached data only, zero API calls.

WHERE THE LAST ROUND LEFT IT
  `pattern` bounds every rule that reads only WHO asserted (+0.25, ns at n=8).
  `lr+rank` clears that ceiling (+1.16 vs count, +0.92 vs ceiling) by reading
  WHERE in its own list each agent put the claim. So the open question is no
  longer "does outside-the-pattern information exist" -- it does -- but how much
  of it we are actually collecting. `lr+rank` spends one linear coefficient per
  agent on log-rank, which is a crude way to spend it.

FOUR NEW ARMS

  nb-rank    Naive Bayes where the likelihood is conditioned on the rank BUCKET,
             not just on the fact of assertion:
                 score = prior + SUM_assert log(TPR_mb / FPR_mb)
                               + SUM_silent log((1-TPR_m.)/(1-FPR_m.))
             Nonparametric in rank, so it does not assume the log-linear decay
             that lr+rank imposes. Omission still contributes.

  pat+rank   The pattern ceiling PLUS the incremental rank evidence:
                 score = logit P(correct|pattern)
                       + SUM_assert log[ (TPR_mb/TPR_m.) / (FPR_mb/FPR_m.) ]
             The bracket is the evidence from WHERE m asserted, given THAT m
             asserted -- so it adds exactly the part the pattern cannot see, and
             nothing else. If any rule should beat both parents, it is this one.

  adapt-*    Per-question keep COUNT instead of a global threshold. This is the
             one structural thing a threshold rule cannot do. Per-question F1 is
                 2*H_k / (k + G)
             with G the number of gold answers, which varies 5..40 across QAMPARI
             questions. So the optimal cut genuinely differs per question, and a
             single global threshold cannot express that however well calibrated
             the score is. Given predicted p_i sorted descending,
                 k*(q) = argmax_k  2 * (sum_{i<=k} p_i) / (k + G_hat_q)
             with G_hat from a ridge regression fitted on the training half.
             Run on top of BOTH the pattern probabilities and the rank
             probabilities, which decomposes adaptivity from rank.

  adapt-*-oracleG   same, with the TRUE gold count. Not a method -- a headroom
             diagnostic. If the oracle is far above the estimate, the gap is in
             predicting G and that is a separate, easier problem.

HONESTY
  5-fold cross-fitted over questions, seed 0, same folds as beyond_pattern.py.
  Every weight, likelihood table, logistic coefficient, ridge coefficient and
  threshold is fitted on 4 folds and scored on the fifth.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v2
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    load, featurize, make_eval, curves, sweep, apply_th, fit_logistic,
    fit_pattern, SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

NB_BUCKETS = 7


def bucket(rank):
    """Rank -> coarse bucket. Log-spaced because precision falls fastest early:
    29% at rank 1, 16% at rank 11, 6% at rank 26."""
    return np.minimum((np.log1p(np.maximum(rank, 0)) / math.log(2)).astype(int),
                      NB_BUCKETS - 1)


def fit_nb_rank(Z, B, y, models):
    """TPR[m,b] = P(m asserts at bucket b | correct), FPR likewise for wrong.
    Returns the per-bucket log-LR, the omission log-LR and the prior."""
    n = len(models)
    pos, neg = y.sum(), len(y) - y.sum()
    tp = np.zeros((n, NB_BUCKETS)); fp = np.zeros((n, NB_BUCKETS))
    for j in range(n):
        a = Z[:, j]
        if not a.any():
            continue
        np.add.at(tp[j], B[a, j], y[a])
        np.add.at(fp[j], B[a, j], 1.0 - y[a])
    sm = 0.5
    TPR = (tp + sm) / (pos + sm * (NB_BUCKETS + 1))
    FPR = (fp + sm) / (neg + sm * (NB_BUCKETS + 1))
    tot_t, tot_f = TPR.sum(1), FPR.sum(1)
    assert_lr = np.log(TPR / FPR)                       # (n, buckets)
    omit_lr = np.log((1 - tot_t) / (1 - tot_f))         # (n,)
    incr_lr = assert_lr - np.log(tot_t / tot_f)[:, None]
    prior = math.log((pos + 1) / (neg + 1))
    return assert_lr, omit_lr, incr_lr, prior


def score_nb(Z, B, assert_lr, omit_lr, prior):
    s = np.full(len(Z), prior)
    for j in range(Z.shape[1]):
        a = Z[:, j]
        s[a] += assert_lr[j][B[a, j]]
        s[~a] += omit_lr[j]
    return s


def score_incr(Z, B, incr_lr):
    s = np.zeros(len(Z))
    for j in range(Z.shape[1]):
        a = Z[:, j]
        s[a] += incr_lr[j][B[a, j]]
    return s


def fit_gold_ridge(feat, target, lam=1.0):
    A = feat.T @ feat + lam * np.eye(feat.shape[1])
    return np.linalg.solve(A, feat.T @ target)


def adaptive_keep(p, rowq, nq, Ghat):
    """Per question keep the top-k that maximises 2*sum(p)/(k+G_hat)."""
    mask = np.zeros(len(p), dtype=bool)
    order = np.lexsort((-p, rowq))
    qs = rowq[order]
    start = np.searchsorted(qs, np.arange(nq), side="left")
    end = np.searchsorted(qs, np.arange(nq), side="right")
    for q in range(nq):
        a, b = start[q], end[q]
        if a >= b:
            continue
        idx = order[a:b]
        cs = np.cumsum(p[idx])
        k = np.arange(1, b - a + 1)
        val = 2 * cs / (k + max(Ghat[q], 1e-6))
        best = int(np.argmax(val))
        if val[best] > 0:
            mask[idx[:best + 1]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="")
    a = ap.parse_args()
    pool = [x for x in a.pool.split(",") if x] or None

    print("=" * 78)
    print("  WEIGHTING v2 -- rank-conditioned likelihoods and per-question budgets")
    print("=" * 78)
    models, qids, cand, ngold, nlist, qtype = load(pool)
    X, y, rowq, pats, rowgi, RK, groups, types = featurize(
        models, qids, cand, nlist, qtype)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, rowgi)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    Z = RK >= 0
    B = bucket(RK)
    cnt = Z.sum(1).astype(float)
    cix = sorted(set(i for g in ("base", "rank") for i in groups[g]))

    spans = []
    for qi in range(nq):
        w = np.nonzero(rowq == qi)[0]
        spans.append((int(w[0]), int(w[-1]) + 1) if len(w) else (0, 0))
    C = lambda s: curves(qids, ngold, rowgi, spans, s)

    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct, gold per question {ng_arr.mean():.1f} "
          f"(range {ng_arr.min():.0f}-{ng_arr.max():.0f})")

    # question-level features for predicting the gold count
    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for qi, q in enumerate(qids):
        w = np.nonzero(rowq == qi)[0]
        QF[qi, 0] = 1.0
        QF[qi, 1] = math.log1p(len(w)) / 5.0
        QF[qi, 2] = (cnt[w].max() if len(w) else 0) / n
        QF[qi, 3] = (cnt[w] >= 2).sum() / 20.0
        QF[qi, 4 + tix.index(qtype[q])] = 1.0

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    ARMS = ["count", "pattern", "lr+rank", "nb-rank", "pat+rank",
            "adapt-count", "adapt-pattern", "adapt-nb", "adapt-rank",
            "adapt-rank-oracleG"]
    per = {a: np.zeros(nq) for a in ARMS}
    ghat_err = []

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        alr, olr, ilr, prior = fit_nb_rank(Z[trm], B[trm], y[trm], models)
        wr = fit_logistic(X[np.ix_(trm, cix)], y[trm])
        lr_s = X[:, cix] @ wr

        scores = {
            "count": cnt,
            "pattern": pat_p,
            "lr+rank": lr_s,
            "nb-rank": score_nb(Z, B, alr, olr, prior),
            "pat+rank": np.log(pat_p / (1 - pat_p)) + score_incr(Z, B, ilr),
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

        # ---- per-question budgets
        beta = fit_gold_ridge(QF[tr_q], ng_arr[tr_q])
        Ghat = np.maximum(QF @ beta, 1.0)
        ghat_err.append(float(np.abs(Ghat[te_q] - ng_arr[te_q]).mean()))
        p_lr = 1.0 / (1.0 + np.exp(-np.clip(lr_s, -30, 30)))
        p_nb = 1.0 / (1.0 + np.exp(-np.clip(scores["nb-rank"], -30, 30)))
        # THE CONTROL THAT MATTERS: budgets driven by P(correct | k) alone. No
        # model identity, no rank -- the only thing it adds to plain counting is
        # that the cut is chosen per question. If this matches adapt-pattern,
        # the gain is adaptivity and reweighting contributes nothing.
        p_cnt = np.array([cr.get(int(c), .05) for c in cnt])
        for arm, p, G in (("adapt-count", p_cnt, Ghat),
                          ("adapt-pattern", pat_p, Ghat),
                          ("adapt-nb", p_nb, Ghat),
                          ("adapt-rank", p_lr, Ghat),
                          ("adapt-rank-oracleG", p_lr, ng_arr)):
            fa, _ = ev(adaptive_keep(p, rowq, nq, G), list(tr_q))
            for qi in te_q:
                per[arm][qi] = fa[qi]
        print(f"    fold {f+1}/{FOLDS} done", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(Z[:, j], list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"  mean |G_hat - G| = {statistics.mean(ghat_err):.2f} answers "
          f"(gold mean {ng_arr.mean():.1f})")
    print(f"\n  {'arm':<20}{'F1':>8}{'vs count':>23}{'vs pattern ceiling':>23}"
          f"{'vs lr+rank':>23}")
    print("  " + "-" * 97)
    ref_c = list(per["count"]); ref_r = list(per["lr+rank"])
    ref_p = list(per["pattern"])
    res = {}

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return f"{d:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if (lo>0 or hi<0) else ' '}"

    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c1 = f"{'(reference)':>23}" if arm == "count" else cell(v, ref_c)
        c2 = f"{'(ceiling)':>23}" if arm == "pattern" else cell(v, ref_p)
        c3 = f"{'(reference)':>23}" if arm == "lr+rank" else cell(v, ref_r)
        print(f"  {arm:<20}{res[arm]:8.2f}{c1:>23}{c2:>23}{c3:>23}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")
    print("  adapt-rank-oracleG uses the true gold count: headroom, not a method")

    json.dump(res, open(RESULTS / "weighting_v2.json", "w"), indent=2)
    print("\n  wrote results/weighting_v2.json")


if __name__ == "__main__":
    main()
