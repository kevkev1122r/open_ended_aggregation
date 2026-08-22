"""
The QAMPARI v5 recipe, ported to ASQA. Does it generalise?

THE RECIPE, as it stands on QAMPARI
  rich per-claim features that the support pattern does not contain
    -- rank of the claim inside each supporter's own output
    -- verbosity: how much each supporter said on this question
    -- OMISSION STRENGTH: how much each SILENT agent said, which is what makes
       its silence weak or strong evidence
    -- the pattern probability, stacked in as a feature
  scored by a gradient boosted model, thresholded with cross-fitted thresholds.

THE TWO THINGS THAT HAVE TO BE DIFFERENT HERE
  LABEL.  analysis/asqa_rank.py fitted "does this cluster cover an
  interpretation", which is recall-flavoured, and produced an arm with the best
  STR-EM and the worst ROUGE-L -- it kept too much and lost under DR. The label
  used here is
      DR(S_i + {i}) > DR(S_i),     S_i = (count>=2 set) \\ {i}
  i.e. does this cluster actually improve the metric.

  UNIT.  A sentence cluster, with semantic rather than exact agreement, and
  "rank" is sentence index inside the author's response.

WHAT THE ANSWER MEANS
  ASQA is the harder case by design: prose, not enumerations; a metric with a
  length penalty; a much more homogeneous agent pool. If the recipe clears the
  pattern ceiling here, it is a general method for open-ended aggregation. If it
  only matches it, the QAMPARI result is about enumerative formats and should be
  claimed that way.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_v5
"""
import json, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.asqa_ensembles import build, cluster
from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.weighting_v2 import bucket, NB_BUCKETS
from open_ended_aggregation.analysis.gbm import GBM, bin_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

SEED = 0
GRID = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.08)
    a = ap.parse_args()

    POOL, qids, per_q, sim, items, refs = build()
    n, nq = len(POOL), len(qids)
    mi = {m: i for i, m in enumerate(POOL)}

    posn, nsent = {}, {}
    for q in qids:
        c = collections.Counter(); p = []
        for m, s in per_q[q]:
            p.append(c[m]); c[m] += 1
        posn[q] = p
        nsent[q] = {m: max(1, c[m]) for m in POOL}

    cl = {q: cluster(per_q[q], sim[q], set(POOL)) for q in qids}
    print(f"  {nq} questions, {n} agents, "
          f"{statistics.mean(len(cl[q]) for q in qids):.1f} clusters/question",
          flush=True)

    D = 2 + 4 * n + 8 + n * NB_BUCKETS
    X, rowq, pats, cover, reps, cidx = [], [], [], [], {}, {}
    for qi, q in enumerate(qids):
        for ci, (ms, rep, gs) in enumerate(cl[q]):
            reps[(qi, ci)] = rep
            cidx[len(X)] = ci
            ps = {}
            for g in gs:
                m = per_q[q][g][0]
                ps[m] = min(ps.get(m, 10 ** 6), posn[q][g])
            x = np.zeros(D); j = 0
            x[j] = 1.0; j += 1
            for m in ms:
                x[j + mi[m]] = 1.0
            j += n
            x[j] = len(ms) / n; j += 1
            pv = [ps[m] for m in ms]
            rv = [ps[m] / nsent[q][m] for m in ms]
            lv = [math.log(nsent[q][m]) for m in ms]
            sil = [m for m in POOL if m not in ms]
            sl = [math.log(nsent[q][m]) for m in sil]
            x[j] = math.log1p(min(pv)) / 3.0
            x[j + 1] = statistics.mean(math.log1p(v) for v in pv) / 3.0
            x[j + 2] = math.log1p(max(pv)) / 3.0
            x[j + 3] = min(rv)
            x[j + 4] = statistics.mean(rv)
            x[j + 5] = statistics.mean(lv) / 3.0
            # OMISSION STRENGTH
            x[j + 6] = (sum(sl) / 3.0) / n if sl else 0.0
            x[j + 7] = (max(sl) / 3.0) if sl else 0.0
            j += 8
            for m in ms:
                x[j + mi[m]] = math.log1p(ps[m]) / 3.0
            j += n
            for m in ms:
                x[j + mi[m]] = math.log(nsent[q][m]) / 3.0
            j += n
            for m in sil:
                x[j + mi[m]] = math.log(nsent[q][m]) / 3.0
            j += n
            for m in ms:
                x[j + mi[m] * NB_BUCKETS + int(bucket(np.array([ps[m]]))[0])] = 1.0
            X.append(x); rowq.append(qi); pats.append(frozenset(ms))
            cover.append(1.0 if A.str_em(rep, items[q]["short_sets"]) > 0 else 0.0)
    X = np.array(X); rowq = np.array(rowq); cover = np.array(cover)
    cnt = np.array([len(p) for p in pats], dtype=float)
    STACK = D - 1

    idx_by_q = collections.defaultdict(list)
    for i, qi in enumerate(rowq):
        idx_by_q[qi].append(i)

    cache = {}

    def dr(qi, keep):
        v = cache.get((qi, keep))
        if v is None:
            txt = " ".join(reps[(qi, c)] for c in keep)
            se = A.str_em(txt, items[qids[qi]]["short_sets"])
            rl = rouge_l(txt, refs[qids[qi]])
            v = cache[(qi, keep)] = (math.sqrt(max(0., se) * max(0., rl)), se, rl)
        return v

    def kept(qi, s, t):
        return tuple(sorted(cidx[i] for i in idx_by_q[qi] if s[i] >= t - 1e-12))

    print("  building the DR-marginal label ...", flush=True)
    base = {qi: set(cidx[i] for i in idx_by_q[qi] if cnt[i] >= 2) for qi in range(nq)}
    drlab = np.zeros(len(X))
    for i in range(len(X)):
        qi, c = rowq[i], cidx[i]
        S = base[qi] - {c}
        drlab[i] = 1.0 if dr(qi, tuple(sorted(S | {c})))[0] > dr(qi, tuple(sorted(S)))[0] else 0.0
    print(f"  {100*cover.mean():.1f}% cover an interpretation, "
          f"{100*drlab.mean():.1f}% improve DR", flush=True)

    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    ARMS = ["count", "weighted", "pattern", "LEAN (cover-label)",
            "LEAN (dr-label)", "GBM (dr-label)"]
    out = {arm: {} for arm in ARMS}
    diag = collections.defaultdict(list)

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
        cr = {k: (x + 1) / (b + 2) for k, (x, b) in bc.items()}
        plut = {p: (x + 20.0 * cr.get(len(p), .3)) / (b + 20.0) for p, (x, b) in bp.items()}
        pat_p = np.array([plut.get(p, cr.get(len(p), .3)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 3.0

        wc = fit_logistic(X[trm], y[trm], ridge=1.0)
        wd = fit_logistic(X[trm], drlab[trm], ridge=1.0)

        tr = np.array(train_q)
        inner, held = tr[: int(0.85 * len(tr))], tr[int(0.85 * len(tr)):]
        im, hm = np.isin(rowq, inner), np.isin(rowq, held)
        _, edges = bin_features(X[trm])
        allc = np.zeros(X.shape, dtype=np.uint8)
        for j, e in enumerate(edges):
            allc[:, j] = np.searchsorted(e, X[:, j], side="right")
        gb = GBM(n_trees=a.trees, lr=a.lr, max_depth=a.depth,
                 subsample=0.8, seed=0).fit(None, drlab[im], codes=allc[im],
                                            edges=edges, val=(allc[hm], drlab[hm]))
        diag["ntrees"].append(len(gb.trees))

        scores = {
            "count": cnt,
            "weighted": np.array([sum(w[m] for m in p) for p in pats]),
            "pattern": pat_p,
            "LEAN (cover-label)": X @ wc,
            "LEAN (dr-label)": X @ wd,
            "GBM (dr-label)": gb.decision(allc),
        }
        for arm, s in scores.items():
            g = np.unique(s[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            best = None
            for t in g:
                v = statistics.mean(dr(qi, kept(qi, s, t))[0] for qi in train_q)
                if best is None or v > best[0]:
                    best = (v, t)
            for qi in test_q:
                out[arm][qi] = dr(qi, kept(qi, s, best[1]))
        print(f"    fold done ({len(cache):,} DR evals, "
              f"{len(gb.trees)} trees)", flush=True)

    singles = {}
    for m in POOL:
        s = np.array([1.0 if m in p else 0.0 for p in pats])
        singles[m] = [dr(qi, kept(qi, s, 1.0))[0] for qi in range(nq)]
    bm = max(POOL, key=lambda m: statistics.mean(singles[m]))
    best_v = singles[bm]
    allq = list(range(nq))
    ref_c = [out["count"][qi][0] for qi in allq]
    ref_p = [out["pattern"][qi][0] for qi in allq]

    print(f"\n  ASQA under DR*   best single = {bm} "
          f"({100*statistics.mean(best_v):.2f})   trees {diag['ntrees']}")
    print(f"  {'arm':<20}{'DR*':>7}{'STR-EM':>8}{'ROUGE-L':>9}"
          f"{'vs best single':>25}{'vs count':>25}{'vs ceiling':>21}")
    print("  " + "-" * 110)

    def cell(v, ref, w=25):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    res = {}
    for arm in ARMS:
        v = [out[arm][qi][0] for qi in allq]
        se = [out[arm][qi][1] for qi in allq]
        rl = [out[arm][qi][2] for qi in allq]
        res[arm] = 100 * statistics.mean(v)
        c1 = f"{'(reference)':>25}" if arm == "count" else cell(v, ref_c)
        if arm == "pattern":
            c2 = f"{'(ceiling)':>21}"
        else:
            d, lo, hi = bootstrap(v, ref_p)
            c2 = f"{d:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<20}{res[arm]:7.2f}{100*statistics.mean(se):8.2f}"
              f"{100*statistics.mean(rl):9.2f}{cell(v, best_v):>25}{c1:>25}{c2:>21}")
    print("\n  DR* = sqrt(ROUGE-L x STR-EM), proxy for published DR")
    json.dump(res, open(RESULTS / "asqa_v5.json", "w"), indent=2)
    print("  wrote results/asqa_v5.json")


if __name__ == "__main__":
    main()
