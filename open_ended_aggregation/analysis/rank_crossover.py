"""
Does the rank mechanism survive the ensemble-size crossover?

THE SETUP
  Two results are now in hand on QAMPARI, and they point opposite ways.

    ensemble_sizes.py   reliability weighting beats counting by +2.25 at n=3 and
                        by -0.36 at n=6. crossover_controls.py showed the same
                        decay under ORACLE weights and under the full
                        support-pattern lookup, so the information available to
                        any who-said-what rule really does shrink as the
                        ensemble grows. Count becomes a sufficient statistic.

    beyond_pattern.py   at n=8 a rule that also reads WHERE the claim sat in
                        each supporter's list beats counting by +1.16 and beats
                        the pattern ceiling by +0.92, both significant. So
                        something still carries information at n=8.

  Those are only compatible if the sufficiency is sufficiency WITHIN the support
  pattern. This file tests that directly: run count, marginal weighting, the
  pattern ceiling and the rank rule over all 210 ensembles of size 3-6 and see
  which of the four curves decay.

  PREDICTION under the reading above: count / weighted / pattern converge as n
  grows (the published crossover), while rank keeps its margin at every size,
  because rank is not a function of who asserted.

PROTOCOL
  2-fold cross-fitted, seed 0, same folds and same 777 questions as
  ensemble_sizes.py, so the count and weighted columns here should reproduce
  that file's numbers. Weights, the logistic coefficients, the pattern lookup
  and the keep threshold are all fitted on the training half only.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.rank_crossover
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import load, fit_logistic
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

SEED = 0


# ------------------------------------------------------------------ tensors
def tensors(models, qids, cand, ngold, nlist):
    """Everything an ensemble needs, as arrays that restrict by column slice."""
    n = len(models)
    mi = {m: i for i, m in enumerate(models)}
    Z, RK, LL, rq, rg = [], [], [], [], []
    for qi, q in enumerate(qids):
        for ms, gi, rk in cand[q]:
            z = np.zeros(n, dtype=bool); r = np.full(n, -1, dtype=np.int32)
            for m in ms:
                z[mi[m]] = True; r[mi[m]] = rk[m]
            Z.append(z); RK.append(r)
            LL.append([math.log(nlist[q][m]) for m in models])
            rq.append(qi); rg.append(-1 if gi is None else gi)
    return (np.array(Z), np.array(RK), np.array(LL, dtype=np.float32),
            np.array(rq), np.array(rg),
            np.array([max(1, ngold[q]) for q in qids], dtype=float))


# ------------------------------------------------------------------ F1 machinery
class Scorer:
    """Per-question F1 of every score threshold, without a Python loop.

    A threshold keeps, per question, the prefix of that question's candidates in
    descending score order. So the F1 of every prefix is computed once by
    cumulative sums, and lowering the threshold adds one candidate at a time --
    which makes the best threshold an argmax over a single cumsum."""

    def __init__(self, rowq, rowgi, ngold, nq):
        self.rowq, self.nq, self.ng = rowq, nq, ngold
        self.ok = rowgi >= 0
        self.span = int(rowgi.max()) + 2 if len(rowgi) else 2
        self.uid = rowq.astype(np.int64) * self.span + rowgi

    def _curve(self, scores):
        o = np.lexsort((-scores, self.rowq))
        qs, oks = self.rowq[o], self.ok[o].astype(float)
        start = np.searchsorted(qs, np.arange(self.nq), side="left")
        s0 = start[qs]
        pos = np.arange(len(qs)) - s0 + 1
        cok = np.cumsum(oks)
        c = cok - np.where(s0 > 0, cok[np.maximum(s0 - 1, 0)], 0.0)
        p_ok = np.nonzero(oks)[0]
        newg = np.zeros(len(qs))
        if len(p_ok):
            _, fi = np.unique(self.uid[o][p_ok], return_index=True)
            newg[p_ok[fi]] = 1.0
        cng = np.cumsum(newg)
        h = cng - np.where(s0 > 0, cng[np.maximum(s0 - 1, 0)], 0.0)
        pr, rc = c / pos, h / self.ng[qs]
        f = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), 0.0)
        prev = np.where(pos > 1, np.concatenate([[0.0], f[:-1]]), 0.0)
        d = np.empty(len(qs)); d[o] = f - prev
        return d

    def best_threshold(self, scores, train_mask_q):
        d = self._curve(scores)
        g = np.argsort(-scores, kind="stable")
        run = np.cumsum(d[g] * train_mask_q[self.rowq[g]])
        if not len(run):
            return 0.0
        # Only positions that CLOSE a tie group are realisable: the rule applied
        # later is `score >= t`, which always takes the whole tied group. Taking
        # the argmax over every position instead lets the sweep score a partial
        # tie group it can never actually keep -- which penalises exactly the
        # arms with the most ties. `count` has only k distinct values, so it is
        # the arm that gets robbed, and the whole table tilts against it.
        s = scores[g]
        edge = np.empty(len(s), dtype=bool)
        edge[:-1] = s[:-1] != s[1:]
        edge[-1] = True
        cand = np.nonzero(edge)[0]
        return float(s[cand[int(np.argmax(run[cand]))]])

    def f1_at(self, scores, t):
        return self.f1_of_mask(scores >= t - 1e-12)

    def f1_of_mask(self, m):
        nk = np.bincount(self.rowq[m], minlength=self.nq).astype(float)
        mh = m & self.ok
        nh = np.bincount(self.rowq[mh], minlength=self.nq).astype(float)
        u = np.unique(self.uid[mh])
        ng = np.bincount((u // self.span).astype(int), minlength=self.nq).astype(float)
        pr = np.divide(nh, nk, out=np.zeros(self.nq), where=nk > 0)
        rc = ng / self.ng
        return np.divide(2 * pr * rc, pr + rc, out=np.zeros(self.nq), where=pr + rc > 0)


# ------------------------------------------------------------------ per-ensemble
def run_ensemble(S, Z, RK, LL, rowq, rowgi, ngold, nq, folds):
    z = Z[:, S]
    keep = z.any(1)
    z = z[keep]
    rk = RK[np.ix_(keep, S)].astype(float)
    ll = LL[np.ix_(keep, S)].astype(float)
    rq, rg = rowq[keep], rowgi[keep]
    sc = Scorer(rq, rg, ngold, nq)
    y = (rg >= 0).astype(float)
    k = len(S)

    cnt = z.sum(1).astype(float)
    code = (z * (1 << np.arange(k))).sum(1)          # support pattern as an int

    lenm = np.maximum(np.exp(ll), 1.0)               # list length of each agent here
    lr_ = np.log1p(np.where(z, rk, 0.0))             # log rank, 0 where silent
    relr = rk / lenm                                 # rank as a fraction of the list
    minr = np.min(np.where(z, rk, np.inf), axis=1)
    F = np.column_stack([
        np.ones(len(z)), z.astype(float), cnt / k,
        np.log1p(minr) / 5.0,
        (lr_.sum(1) / np.maximum(cnt, 1)) / 5.0,
        np.min(np.where(z, relr, np.inf), axis=1),
        np.where(z, relr, 0.0).sum(1) / np.maximum(cnt, 1),
        lr_ / 5.0,
    ])

    arms = {}
    for train_q, test_q in folds:
        tm = np.zeros(nq, dtype=bool); tm[train_q] = True
        trm = tm[rq]

        # marginal reliability, as in ensemble_sizes.py
        w = np.array([( (y[trm] * z[trm, j]).sum() + 1) / (z[trm, j].sum() + 2)
                      for j in range(k)])
        # shrunk P(correct | exact pattern), as in crossover_controls.py
        nb = np.bincount(code[trm], weights=y[trm], minlength=1 << k)
        nt = np.bincount(code[trm], minlength=1 << k)
        pc = np.bincount(cnt[trm].astype(int), weights=y[trm], minlength=k + 1)
        pt = np.bincount(cnt[trm].astype(int), minlength=k + 1)
        cr = (pc + 1) / (pt + 2)
        plut = (nb + 20.0 * cr[popcount(np.arange(1 << k))]) / (nt + 20.0)

        wr = fit_logistic(F[trm], y[trm])

        s_rank = F @ wr
        for arm, s in (("count", cnt),
                       ("weighted", z @ w),
                       ("pattern", plut[code]),
                       ("rank", s_rank)):
            t = sc.best_threshold(s, tm)
            f = sc.f1_at(s, t)
            arms.setdefault(arm, np.zeros(nq))[test_q] = f[test_q]

        # rank + per-question budget: the champion's decision rule, restricted
        # to this ensemble. Ghat is fitted on the training half only.
        pr = 1.0 / (1.0 + np.exp(-np.clip(s_rank, -30, 30)))
        QG = np.zeros((nq, 5))
        for qi in range(nq):
            ww = np.nonzero(rq == qi)[0]
            if not len(ww):
                QG[qi] = [1.0, 0, 0, 0, 0]
                continue
            QG[qi] = [1.0, np.log1p(len(ww)) / 5.0, cnt[ww].max() / k,
                      (cnt[ww] >= 2).sum() / 20.0, pr[ww].sum() / 10.0]
        beta = fit_gold_ridge(QG[train_q], ngold[train_q])
        Gh = np.maximum(QG @ beta, 1.0)
        fb = sc.f1_of_mask(adaptive_keep(pr, rq, nq, Gh))
        arms.setdefault("rank+budget", np.zeros(nq))[test_q] = fb[test_q]
    return arms


def popcount(a):
    a = a.astype(np.uint32)
    out = np.zeros(len(a), dtype=int)
    for b in range(32):
        out += (a >> b) & 1
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3,4,5,6")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    models, qids, cand, ngold, nlist, qtype = load()
    Z, RK, LL, rowq, rowgi, ng = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    print(f"  QAMPARI  {nq} questions, pool of {len(models)}, "
          f"{len(rowq):,} candidate claims", flush=True)

    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    singles = {}
    sc_full = Scorer(rowq, rowgi, ng, nq)
    for j, m in enumerate(models):
        s = Z[:, j].astype(float)
        singles[j] = sc_full.f1_at(s, 1.0)

    out = []
    for k in sizes:
        for S in itertools.combinations(range(len(models)), k):
            arms = run_ensemble(list(S), Z, RK, LL, rowq, rowgi, ng, nq, folds)
            bj = max(S, key=lambda j: singles[j].mean())
            row = dict(k=k, ens=[models[j] for j in S], best_model=models[bj],
                       best=100 * singles[bj].mean())
            for arm, v in arms.items():
                row[arm] = 100 * v.mean()
            for arm in ("weighted", "pattern", "rank", "rank+budget"):
                d, lo, hi = bootstrap(list(arms[arm]), list(arms["count"]))
                row[f"{arm}_vs_cnt"] = d
                row[f"{arm}_sig"] = bool(lo > 0 or hi < 0)
            pr = sorted((singles[j].mean() for j in S), reverse=True)
            row["dominance"] = pr[0] / max(1e-9, statistics.mean(pr[1:]))
            out.append(row)
        print(f"    size {k}: {sum(1 for r in out if r['k']==k)} ensembles done",
              flush=True)

    print(f"\n  GAIN OVER COUNTING, mean across ensembles of each size")
    print(f"  {'size':>5}{'n':>5}{'best single':>13}{'count':>9}"
          f"{'weighted':>11}{'pattern':>10}{'rank':>10}{'rank+budget':>14}")
    print("  " + "-" * 77)
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        c = statistics.mean(r["count"] for r in R)
        W = {"weighted": 11, "pattern": 10, "rank": 10, "rank+budget": 14}
        print(f"  {k:>5}{len(R):>5}{statistics.mean(r['best'] for r in R):>13.2f}"
              f"{c:>9.2f}" + "".join(
                  f"{statistics.mean(r[x] for r in R) - c:>+{W[x]}.2f}"
                  for x in W))

    print(f"\n  ensembles where the arm SIGNIFICANTLY beats counting")
    print(f"  {'size':>5}{'weighted':>12}{'pattern':>12}{'rank':>12}{'rank+budget':>14}")
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        cells = "".join(
            f"{sum(1 for r in R if r[f'{x}_sig'] and r[f'{x}_vs_cnt'] > 0):>7}/{len(R):<5}"
            for x in ("weighted", "pattern", "rank", "rank+budget"))
        print(f"  {k:>5}{cells}")

    json.dump(out, open(RESULTS / "rank_crossover.json", "w"), indent=1)
    print("\n  wrote results/rank_crossover.json")


if __name__ == "__main__":
    main()
