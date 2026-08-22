"""
Does the QAMPARI v2 story transfer to ASQA? Two things under test.

1. PER-QUESTION BUDGETS
   On QAMPARI, choosing the keep COUNT per question rather than thresholding a
   global score is worth +1.19 over counting **with no model identity at all**
   (`adapt-count` in analysis/weighting_v2.py), and +1.97 when combined with
   rank. If that transfers, it is the most portable finding in the project,
   because it needs no reliability estimates of any kind.

   ASQA analogue: keep the top-k clusters by predicted coverage, with
       k*(q) = argmax_k  2 * (sum_{i<=k} p_i) / (k + G_hat_q)
   and G_hat the predicted number of interpretations. The surrogate is
   F1-shaped between coverage and length, which is the tradeoff DR makes.
   The arms are still SCORED on true DR*, never on the surrogate.

2. THE LABEL FIX
   analysis/asqa_rank.py fitted the logistic to "does this cluster cover an
   interpretation", which is recall-flavoured, and the arm came out with the
   best STR-EM (47.76) and the worst ROUGE-L (19.99) -- it kept too much, and
   landed 0.90 BELOW the pattern ceiling. DR penalises length; the label did
   not. So that arm was never a fair test of whether rank generalises.

   `dr-label` fits the same features to a label that knows about length:
       label_i = DR(S_i + {i})  >  DR(S_i),     S_i = (count>=2 set) \\ {i}
   i.e. does including this cluster actually improve DR. Two DR evaluations per
   cluster, computed once, cached.

   If `lr+rank(dr-label)` clears the ceiling, the ASQA negative was a
   methodology artefact and the rank mechanism is general. If it still does not,
   position in prose really does carry nothing and the claim is
   enumerative-only. Either answer is publishable; the current one is not,
   because it confounds the two.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_rank2
"""
import json, math, random, collections, statistics, itertools

import numpy as np

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.asqa_ensembles import build, cluster
from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.weighting_v2 import (
    bucket, fit_nb_rank, score_nb, fit_gold_ridge, NB_BUCKETS)
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

SEED = 0
GRID = 160


def main():
    POOL, qids, per_q, sim, items, refs = build()
    n = len(POOL)
    mi = {m: i for i, m in enumerate(POOL)}
    nq = len(qids)

    posn, nsent = {}, {}
    for q in qids:
        c = collections.Counter(); p = []
        for m, s in per_q[q]:
            p.append(c[m]); c[m] += 1
        posn[q] = p
        nsent[q] = {m: max(1, c[m]) for m in POOL}

    cl = {q: cluster(per_q[q], sim[q], set(POOL)) for q in qids}
    print(f"  mean clusters/question {statistics.mean(len(cl[q]) for q in qids):.1f}",
          flush=True)

    rows, rowq, pats, cover, reps, RKm = [], [], [], [], {}, []
    for qi, q in enumerate(qids):
        for ci, (ms, rep, gs) in enumerate(cl[q]):
            reps[(qi, ci)] = rep
            ps = {}
            for g in gs:
                m = per_q[q][g][0]
                ps[m] = min(ps.get(m, 10 ** 6), posn[q][g])
            x = np.zeros(2 + 2 * n + 6); j = 0
            x[j] = 1.0; j += 1
            for m in ms:
                x[j + mi[m]] = 1.0
            j += n
            x[j] = len(ms) / n; j += 1
            pv = [ps[m] for m in ms]
            rv = [ps[m] / nsent[q][m] for m in ms]
            lv = [math.log(nsent[q][m]) for m in ms]
            x[j] = math.log1p(min(pv)) / 3.0
            x[j + 1] = statistics.mean(math.log1p(v) for v in pv) / 3.0
            x[j + 2] = min(rv); x[j + 3] = statistics.mean(rv)
            x[j + 4] = statistics.mean(lv) / 3.0
            x[j + 5] = min(lv) / 3.0
            j += 6
            for m in ms:
                x[j + mi[m]] = math.log1p(ps[m]) / 3.0
            rk = np.full(n, -1)
            for m in ms:
                rk[mi[m]] = ps[m]
            rows.append(x); rowq.append(qi); pats.append(frozenset(ms))
            RKm.append(rk)
            cover.append(1.0 if A.str_em(rep, items[q]["short_sets"]) > 0 else 0.0)
    X = np.array(rows); rowq = np.array(rowq); cover = np.array(cover)
    RKm = np.array(RKm); Zm = RKm >= 0; Bm = bucket(RKm)
    cnt = np.array([len(p) for p in pats], dtype=float)
    words = np.array([len(reps[(rowq[i], c)].split())
                      for i, c in enumerate(_cidx(rowq))], dtype=float)

    idx_by_q = collections.defaultdict(list)
    for i, qi in enumerate(rowq):
        idx_by_q[qi].append(i)
    cidx = {}
    for qi, ids in idx_by_q.items():
        for c, i in enumerate(sorted(ids)):
            cidx[i] = c

    cache = {}

    def dr(qi, keep_ids):
        key = (qi, keep_ids)
        v = cache.get(key)
        if v is None:
            txt = " ".join(reps[(qi, c)] for c in keep_ids)
            se = A.str_em(txt, items[qids[qi]]["short_sets"])
            rl = rouge_l(txt, refs[qids[qi]])
            v = cache[key] = (math.sqrt(max(0.0, se) * max(0.0, rl)), se, rl)
        return v

    def kept_th(qi, s, t):
        return tuple(cidx[i] for i in sorted(idx_by_q[qi]) if s[i] >= t - 1e-12)

    def kept_mask(qi, mask):
        return tuple(cidx[i] for i in sorted(idx_by_q[qi]) if mask[i])

    # ---- the length-aware label, computed once
    print("  building the DR-marginal label ...", flush=True)
    base = {qi: set(cidx[i] for i in idx_by_q[qi] if cnt[i] >= 2) for qi in range(nq)}
    drlab = np.zeros(len(X))
    for i in range(len(X)):
        qi, c = rowq[i], cidx[i]
        S = base[qi] - {c}
        with_i = dr(qi, tuple(sorted(S | {c})))[0]
        without = dr(qi, tuple(sorted(S)))[0]
        drlab[i] = 1.0 if with_i > without else 0.0
    print(f"  {100*cover.mean():.1f}% of clusters cover an interpretation; "
          f"{100*drlab.mean():.1f}% actually improve DR", flush=True)

    # ---- question features for the budget
    ninterp = np.array([float(items[q].get("n_interp") or len(items[q]["short_sets"]))
                        for q in qids])
    QF = np.zeros((nq, 4))
    for qi in range(nq):
        w = np.array(idx_by_q[qi])
        QF[qi] = [1.0, math.log1p(len(w)) / 3.0,
                  cnt[w].max() / n, (cnt[w] >= 2).sum() / 10.0]

    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    ARMS = ["count", "weighted", "pattern", "lr+rank", "lr+rank(dr-label)",
            "nb-rank", "adapt-count", "adapt-pattern", "adapt-rank(dr-label)",
            "learned-budget(nb-rank)", "learned-budget(dr-label)"]
    out = {a: {} for a in ARMS}

    for train_q, test_q in folds:
        tm = np.zeros(nq, dtype=bool); tm[train_q] = True
        trm = tm[rowq]
        y = cover

        w = {m: ((y[trm] * np.array([m in p for p in pats])[trm]).sum() + 1) /
                (np.array([m in p for p in pats])[trm].sum() + 2) for m in POOL}
        bp = collections.defaultdict(lambda: [0, 0]); bc = collections.defaultdict(lambda: [0, 0])
        for p, ok, t in zip(pats, y, trm):
            if not t:
                continue
            bp[p][0] += ok; bp[p][1] += 1
            bc[len(p)][0] += ok; bc[len(p)][1] += 1
        cr = {k: (a + 1) / (b + 2) for k, (a, b) in bc.items()}
        plut = {p: (a + 20.0 * cr.get(len(p), .3)) / (b + 20.0) for p, (a, b) in bp.items()}
        pat_p = np.array([plut.get(p, cr.get(len(p), .3)) for p in pats])

        wr = fit_logistic(X[trm], y[trm])
        wd = fit_logistic(X[trm], drlab[trm])
        alr, olr, ilr, prior = fit_nb_rank(Zm[trm], Bm[trm], y[trm], POOL)

        scores = {
            "count": cnt,
            "weighted": np.array([sum(w[m] for m in p) for p in pats]),
            "pattern": pat_p,
            "lr+rank": X @ wr,
            "lr+rank(dr-label)": X @ wd,
            "nb-rank": score_nb(Zm, Bm, alr, olr, prior),
        }
        for arm, s in scores.items():
            g = np.unique(s[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            best = None
            for t in g:
                v = statistics.mean(dr(qi, kept_th(qi, s, t))[0] for qi in train_q)
                if best is None or v > best[0]:
                    best = (v, t)
            for qi in test_q:
                out[arm][qi] = dr(qi, kept_th(qi, s, best[1]))

        # ---- per-question budgets
        beta = fit_gold_ridge(QF[train_q], ninterp[train_q])
        Gh = np.maximum(QF @ beta, 1.0)
        p_cnt = np.array([cr.get(int(c), .3) for c in cnt])
        p_dr = 1.0 / (1.0 + np.exp(-np.clip(X @ wd, -30, 30)))
        for arm, p in (("adapt-count", p_cnt), ("adapt-pattern", pat_p),
                       ("adapt-rank(dr-label)", p_dr)):
            mask = np.zeros(len(p), dtype=bool)
            for qi in range(nq):
                ids = np.array(sorted(idx_by_q[qi], key=lambda i: -p[i]))
                if not len(ids):
                    continue
                cs = np.cumsum(p[ids]); k = np.arange(1, len(ids) + 1)
                val = 2 * cs / (k + Gh[qi])
                mask[ids[:int(np.argmax(val)) + 1]] = True
            for qi in test_q:
                out[arm][qi] = dr(qi, kept_mask(qi, mask))

        # ---- budget LEARNED against the real metric, not against a surrogate.
        # The plug-in rule above assumes per-question F1's shape, 2S/(k+G).
        # DR is sqrt(STR-EM x ROUGE-L) and ROUGE-L collapses with length far
        # faster than 1/(k+G), so that surrogate keeps far too much here. This
        # instead finds the DR-optimal k on the TRAINING questions and regresses
        # it on question features -- metric-agnostic, and it ports to any scorer.
        for arm, p in (("learned-budget(nb-rank)", scores["nb-rank"]),
                       ("learned-budget(dr-label)", X @ wd)):
            ordq = {qi: sorted(idx_by_q[qi], key=lambda i: -p[i]) for qi in range(nq)}
            QF2 = np.zeros((nq, 6))
            for qi in range(nq):
                w_ = np.array(idx_by_q[qi])
                QF2[qi] = [1.0, math.log1p(len(w_)) / 3.0, cnt[w_].max() / n,
                           (cnt[w_] >= 2).sum() / 10.0,
                           float(np.mean(p[w_])), float(np.max(p[w_]))]
            kstar = []
            for qi in train_q:
                ids = ordq[qi]
                best = (0, -1.0)
                for k in range(len(ids) + 1):
                    v = dr(qi, tuple(sorted(cidx[i] for i in ids[:k])))[0]
                    if v > best[1]:
                        best = (k, v)
                kstar.append(best[0])
            b2 = fit_gold_ridge(QF2[train_q], np.array(kstar, dtype=float))
            khat = np.clip(np.round(QF2 @ b2), 0, None).astype(int)
            for qi in test_q:
                ids = ordq[qi][:khat[qi]]
                out[arm][qi] = dr(qi, tuple(sorted(cidx[i] for i in ids)))
        print(f"    fold done ({len(cache):,} DR evals cached)", flush=True)

    singles = {}
    for m in POOL:
        s = np.array([1.0 if m in p else 0.0 for p in pats])
        singles[m] = [dr(qi, kept_th(qi, s, 1.0))[0] for qi in range(nq)]
    bm = max(POOL, key=lambda m: statistics.mean(singles[m]))
    allq = list(range(nq))
    ref_c = [out["count"][qi][0] for qi in allq]
    ref_p = [out["pattern"][qi][0] for qi in allq]

    print(f"\n  ASQA under DR*   best single = {bm} "
          f"({100*statistics.mean(singles[bm]):.2f})")
    print(f"  {'arm':<22}{'DR*':>7}{'STR-EM':>8}{'ROUGE-L':>9}"
          f"{'vs count':>22}{'vs pattern ceiling':>22}")
    print("  " + "-" * 90)
    res = {}
    for arm in ARMS:
        v = [out[arm][qi][0] for qi in allq]
        se = [out[arm][qi][1] for qi in allq]
        rl = [out[arm][qi][2] for qi in allq]
        res[arm] = 100 * statistics.mean(v)
        if arm == "count":
            c1 = f"{'(reference)':>22}"
        else:
            d, lo, hi = bootstrap(v, ref_c)
            c1 = f"{d:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        if arm == "pattern":
            c2 = f"{'(ceiling)':>22}"
        else:
            d, lo, hi = bootstrap(v, ref_p)
            c2 = f"{d:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<22}{res[arm]:7.2f}{100*statistics.mean(se):8.2f}"
              f"{100*statistics.mean(rl):9.2f}{c1:>22}{c2:>22}")
    print("\n  DR* = sqrt(ROUGE-L x STR-EM), proxy for published DR")
    json.dump(res, open(RESULTS / "asqa_rank2.json", "w"), indent=2)
    print("  wrote results/asqa_rank2.json")


def _cidx(rowq):
    """cluster index within its question, in row order"""
    seen = collections.Counter()
    out = []
    for qi in rowq:
        out.append(seen[qi]); seen[qi] += 1
    return out


if __name__ == "__main__":
    main()
