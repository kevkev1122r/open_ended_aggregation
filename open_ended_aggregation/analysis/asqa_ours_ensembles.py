"""
Our method on every ASQA ensemble, alongside counting and reliability weighting.

WHY THIS EXISTS
  analysis/asqa_ensembles.py sweeps all 42 ASQA ensembles but only for `count`
  and `weighted`. The per-ensemble table therefore had two of the five arms, and
  our own method was missing from it -- the first thing a reader checks.

  This adds the rank scorer. The per-question BUDGET is deliberately not included:
  it was measured on the full ASQA pool and does not transfer (adapt-count +0.22
  ns, every budget variant below the pattern ceiling), because DR's length
  penalty is steeper than the F1-shaped surrogate the budget rule assumes. So
  what is swept here is the scoring half of the method only, and the table says
  so.

COST, AND THE TRICK THAT MAKES IT FEASIBLE
  Every candidate threshold needs a ROUGE-L pass over 400 questions, and a
  continuous score has as many thresholds as clusters. Scoring naively would be
  ~1.5M DR evaluations across 42 ensembles.

  Instead: a threshold keeps, per question, a PREFIX of the clusters in
  descending score order. So sorting once and walking the prefix prices every
  threshold in one pass -- the same trick used in analysis/asc_ensembles.py.
  DR is memoised on the exact kept-set, which recurs heavily across ensembles.

LABEL
  Clusters are labelled by whether the representative covers an interpretation.
  The DR-marginal label used in analysis/asqa_rank2.py is better motivated but
  costs two DR evaluations per cluster per ensemble; on the full pool the two
  labels differ by +0.07 (26.88 vs 26.81), which does not justify 42x the cost.
  Disclosed rather than quietly swapped.

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_ours_ensembles [--sizes 3]
"""
import json, math, time, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.asqa_ensembles import build, cluster, sentences
from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.weighting_v2 import bucket, NB_BUCKETS
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3,4,5,6")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    POOL, qids, per_q, sim, items, refs = build()
    n, nq = len(POOL), len(qids)
    mi = {m: i for i, m in enumerate(POOL)}

    # sentence index inside its author's response, and that author's length
    posn, nsent = {}, {}
    for q in qids:
        c = collections.Counter(); p = []
        for m, s in per_q[q]:
            p.append(c[m]); c[m] += 1
        posn[q] = p
        nsent[q] = {m: max(1, c[m]) for m in POOL}

    cache = {}

    def dr(qi, sids):
        v = cache.get((qi, sids))
        if v is None:
            txt = " ".join(per_q[qids[qi]][i][1] for i in sids)
            se = A.str_em(txt, items[qids[qi]]["short_sets"])
            rl = rouge_l(txt, refs[qids[qi]])
            v = cache[(qi, sids)] = math.sqrt(max(0., se) * max(0., rl))
        return v

    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    out = []
    for k in sizes:
        t0 = time.time()
        for ens in itertools.combinations(POOL, k):
            E = set(ens)
            cl = {q: cluster(per_q[q], sim[q], E) for q in qids}

            # per-cluster: features, cover label, representative sentence id
            feats, cover, rep_id, byq_idx = {}, {}, {}, {}
            D = 2 + 3 * k + 6
            for qi, q in enumerate(qids):
                rows = []
                for ci, (ms, rep, gs) in enumerate(cl[q]):
                    ps = {}
                    for g in gs:
                        m = per_q[q][g][0]
                        ps[m] = min(ps.get(m, 10 ** 6), posn[q][g])
                    x = np.zeros(D); j = 0
                    x[j] = 1.0; j += 1
                    for m in ms:
                        x[j + mi[m] % k if False else j + list(ens).index(m)] = 1.0
                    j += k
                    x[j] = len(ms) / k; j += 1
                    pv = [ps[m] for m in ms]
                    rv = [ps[m] / nsent[q][m] for m in ms]
                    sil = [m for m in ens if m not in ms]
                    sl = [math.log(nsent[q][m]) for m in sil]
                    x[j] = math.log1p(min(pv)) / 3.0
                    x[j + 1] = statistics.mean(math.log1p(v) for v in pv) / 3.0
                    x[j + 2] = min(rv)
                    x[j + 3] = statistics.mean(rv)
                    x[j + 4] = (sum(sl) / 3.0) / k if sl else 0.0     # omission
                    x[j + 5] = (max(sl) / 3.0) if sl else 0.0
                    j += 6
                    for m in ms:
                        x[j + list(ens).index(m)] = math.log1p(ps[m]) / 3.0
                    j += k
                    for m in sil:
                        x[j + list(ens).index(m)] = math.log(nsent[q][m]) / 3.0
                    rows.append((x, frozenset(ms),
                                 1.0 if A.str_em(rep, items[q]["short_sets"]) > 0 else 0.0,
                                 max(gs, key=lambda g: len(per_q[q][g][1]))))
                byq_idx[qi] = rows

            allX = np.array([r[0] for qi in range(nq) for r in byq_idx[qi]])
            ally = np.array([r[2] for qi in range(nq) for r in byq_idx[qi]])
            allq = np.array([qi for qi in range(nq) for _ in byq_idx[qi]])
            pats = [r[1] for qi in range(nq) for r in byq_idx[qi]]
            sids = [r[3] for qi in range(nq) for r in byq_idx[qi]]
            cnt = np.array([len(p) for p in pats], dtype=float)

            def curve(score):
                """DR at every prefix, per question, in one pass."""
                S, F = [], []
                for qi in range(nq):
                    w = np.nonzero(allq == qi)[0]
                    if not len(w):
                        S.append(np.array([])); F.append(np.array([0.0])); continue
                    o = w[np.argsort(-score[w], kind="stable")]
                    f = [0.0]
                    for j in range(1, len(o) + 1):
                        f.append(dr(qi, tuple(sorted(sids[i] for i in o[:j]))))
                    S.append(-score[o]); F.append(np.array(f))
                return S, F

            def best_th(S, F, qs, grid):
                tot = np.zeros(len(grid)); neg = -grid
                for qi in qs:
                    if not len(S[qi]):
                        continue
                    tot += F[qi][np.searchsorted(S[qi], neg, side="right")]
                return grid[int(np.argmax(tot))]

            def at(S, F, qi, t):
                if not len(S[qi]):
                    return 0.0
                return float(F[qi][int(np.searchsorted(S[qi], -t, side="right"))])

            scored = {aa: np.zeros(nq) for aa in ("count", "weighted", "ours")}
            for tr, te in folds:
                trm = np.isin(allq, tr)
                w_ = {m: ((ally[trm] * np.array([m in p for p in pats])[trm]).sum() + 1) /
                         (np.array([m in p for p in pats])[trm].sum() + 2) for m in ens}
                wl = fit_logistic(allX[trm], ally[trm], ridge=1.0)
                arms = {"count": cnt,
                        "weighted": np.array([sum(w_[m] for m in p) for p in pats]),
                        "ours": allX @ wl}
                for aa, sc in arms.items():
                    S, F = curve(sc)
                    g = np.unique(sc[trm])
                    if len(g) > 120:
                        g = np.quantile(g, np.linspace(0, 1, 120))
                    t = best_th(S, F, tr, g)
                    for qi in te:
                        scored[aa][qi] = at(S, F, qi, t)

            singles = {}
            for m in ens:
                sc = np.array([1.0 if m in p else 0.0 for p in pats])
                S, F = curve(sc)
                singles[m] = np.array([at(S, F, qi, 1.0) for qi in range(nq)])
            bm = max(ens, key=lambda m: singles[m].mean())
            d1, l1, h1 = bootstrap(list(scored["weighted"]), list(scored["count"]))
            d2, l2, h2 = bootstrap(list(scored["ours"]), list(scored["count"]))
            out.append(dict(k=k, ens=list(ens), best_model=bm,
                            best=100 * singles[bm].mean(),
                            count=100 * scored["count"].mean(),
                            weighted=100 * scored["weighted"].mean(),
                            ours=100 * scored["ours"].mean(),
                            wgt_vs_cnt=d1, wgt_sig=bool(l1 > 0 or h1 < 0),
                            ours_vs_cnt=d2, ours_sig=bool(l2 > 0 or h2 < 0)))
        print(f"    size {k}: {sum(1 for r in out if r['k']==k)} ensembles "
              f"({time.time()-t0:.0f}s, {len(cache):,} DR evals cached)", flush=True)

    print(f"\n  ASQA under DR*  --  mean over ensembles by size")
    print(f"  {'n':>4}{'ens':>6}{'best single':>13}{'count':>9}{'weighted':>10}"
          f"{'ours':>9}{'ours-cnt':>11}{'ours as %':>11}")
    print("  " + "-" * 73)
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        if not R:
            continue
        c = statistics.mean(r["count"] for r in R)
        o = statistics.mean(r["ours"] for r in R)
        print(f"  {k:>4}{len(R):>6}{statistics.mean(r['best'] for r in R):>13.2f}"
              f"{c:>9.2f}{statistics.mean(r['weighted'] for r in R):>10.2f}"
              f"{o:>9.2f}{o-c:>+11.2f}{100*(o-c)/c:>+10.1f}%")

    print(f"\n  ensembles significantly beating counting")
    print(f"  {'n':>4}{'weighted':>12}{'ours':>12}")
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        if not R:
            continue
        x = sum(1 for r in R if r["wgt_sig"] and r["wgt_vs_cnt"] > 0)
        y = sum(1 for r in R if r["ours_sig"] and r["ours_vs_cnt"] > 0)
        print(f"  {k:>4}{f'{x}/{len(R)}':>12}{f'{y}/{len(R)}':>12}")

    json.dump(out, open(RESULTS / "asqa_ours_ensembles.json", "w"), indent=1)
    print("\n  wrote results/asqa_ours_ensembles.json")


if __name__ == "__main__":
    main()
