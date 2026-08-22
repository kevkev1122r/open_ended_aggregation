"""
Does the rank mechanism generalise, or is it a QAMPARI artefact?

WHAT IT HAS TO SURVIVE
  On QAMPARI, a rule that reads WHERE a claim sat in each agent's list beats the
  support-pattern ceiling by +0.92 (analysis/beyond_pattern.py). QAMPARI answers
  are literally numbered lists, so "position" is unambiguous and the model was
  arguably ordering by its own confidence. That is the cheap explanation, and it
  would not generalise.

  ASQA is the hard case for it. The atomic unit is a SENTENCE CLUSTER, not a
  list entry. Agents write prose, not enumerations. Position in a paragraph is
  rhetorical structure, not a ranked list. The metric is DR*, which has a length
  penalty, rather than set F1. If a position feature still clears the pattern
  ceiling here, the mechanism is about how models allocate confidence within
  their own output, and it belongs in the paper as a general claim.

  ANALOGUES USED
    rank       index of the sentence inside its author's response
    verbosity  how many sentences that author wrote for this question

ARMS   count | weighted(marginal) | pattern(ceiling) | lr+rank
  Same family as analysis/weighting_schemes.py so the ASQA column there
  (count 23.98, marginal 27.45, pattern 27.37) is directly comparable.

  2-fold cross-fitted rather than 5 -- every threshold evaluation costs a
  ROUGE-L pass over 400 questions, so folds are the expensive axis here.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_rank
"""
import json, math, random, collections, statistics, itertools

import numpy as np

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.asqa_ensembles import build, cluster
from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS

SEED = 0
GRID = 160


def main():
    POOL, qids, per_q, sim, items, refs = build()
    n = len(POOL)
    mi = {m: i for i, m in enumerate(POOL)}

    # sentence index within its author's response, and that author's length
    posn, nsent = {}, {}
    for q in qids:
        c = collections.Counter()
        p = []
        for m, s in per_q[q]:
            p.append(c[m]); c[m] += 1
        posn[q] = p
        nsent[q] = {m: max(1, c[m]) for m in POOL}

    cl = {q: cluster(per_q[q], sim[q], set(POOL)) for q in qids}
    print(f"  mean clusters/question {statistics.mean(len(cl[q]) for q in qids):.1f}",
          flush=True)

    # per-cluster features and the fitting label
    rows, rowq, y, pats = [], [], [], []
    reps = {}
    for qi, q in enumerate(qids):
        for ci, (ms, rep, gs) in enumerate(cl[q]):
            reps[(qi, ci)] = rep
            ps = {}
            for g in gs:
                m = per_q[q][g][0]
                ps[m] = min(ps.get(m, 10 ** 6), posn[q][g])
            x = np.zeros(2 + 2 * n + 6)
            j = 0
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
            x[j + 2] = min(rv)
            x[j + 3] = statistics.mean(rv)
            x[j + 4] = statistics.mean(lv) / 3.0
            x[j + 5] = min(lv) / 3.0
            j += 6
            for m in ms:
                x[j + mi[m]] = math.log1p(ps[m]) / 3.0
            rows.append(x); rowq.append(qi); pats.append(frozenset(ms))
            y.append(1.0 if A.str_em(rep, items[q]["short_sets"]) > 0 else 0.0)
    X = np.array(rows); y = np.array(y); rowq = np.array(rowq)
    print(f"  {len(y):,} clusters, {100*y.mean():.1f}% cover an interpretation",
          flush=True)

    idx_by_q = collections.defaultdict(list)
    for i, qi in enumerate(rowq):
        idx_by_q[qi].append(i)

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

    # cluster index within its question, aligned with the row index
    cidx = {}
    for qi, ids in idx_by_q.items():
        for c, i in enumerate(sorted(ids)):
            cidx[i] = c

    def kept(qi, scores, t):
        return tuple(cidx[i] for i in sorted(idx_by_q[qi]) if scores[i] >= t - 1e-12)

    rng = random.Random(SEED)
    order = list(range(len(qids))); rng.shuffle(order)
    h = len(order) // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    ARMS = ["count", "weighted", "pattern", "lr+rank"]
    out = {a: {} for a in ARMS}
    for train_q, test_q in folds:
        tm = np.zeros(len(qids), dtype=bool); tm[train_q] = True
        trm = tm[rowq]
        cnt = np.array([len(p) for p in pats], dtype=float)

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
        wr = fit_logistic(X[trm], y[trm])

        scores = {
            "count": cnt,
            "weighted": np.array([sum(w[m] for m in p) for p in pats]),
            "pattern": np.array([plut.get(p, cr.get(len(p), .3)) for p in pats]),
            "lr+rank": X @ wr,
        }
        for arm in ARMS:
            s = scores[arm]
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
        print(f"    fold done ({len(cache):,} DR evals cached)", flush=True)

    singles = {}
    for m in POOL:
        s = np.array([1.0 if m in p else 0.0 for p in pats])
        singles[m] = [dr(qi, kept(qi, s, 1.0))[0] for qi in range(len(qids))]
    bm = max(POOL, key=lambda m: statistics.mean(singles[m]))
    best = singles[bm]
    allq = list(range(len(qids)))
    cnt_v = [out["count"][qi][0] for qi in allq]
    pat_v = [out["pattern"][qi][0] for qi in allq]

    print(f"\n  ASQA under DR*   best single = {bm} ({100*statistics.mean(best):.2f})")
    print(f"  {'arm':<12}{'DR*':>8}{'STR-EM':>9}{'ROUGE-L':>9}"
          f"{'vs count':>22}{'vs pattern ceiling':>22}")
    print("  " + "-" * 82)
    res = {}
    for arm in ARMS:
        v = [out[arm][qi][0] for qi in allq]
        se = [out[arm][qi][1] for qi in allq]
        rl = [out[arm][qi][2] for qi in allq]
        res[arm] = 100 * statistics.mean(v)
        if arm == "count":
            c1 = f"{'(reference)':>22}"
        else:
            d, lo, hi = bootstrap(v, cnt_v)
            c1 = f"{d:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        if arm == "pattern":
            c2 = f"{'(ceiling)':>22}"
        else:
            d, lo, hi = bootstrap(v, pat_v)
            c2 = f"{d:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<12}{res[arm]:8.2f}{100*statistics.mean(se):9.2f}"
              f"{100*statistics.mean(rl):9.2f}{c1:>22}{c2:>22}")
    print("\n  DR* = sqrt(ROUGE-L x STR-EM), proxy for published DR")
    json.dump(res, open(RESULTS / "asqa_rank.json", "w"), indent=2)
    print("  wrote results/asqa_rank.json")


if __name__ == "__main__":
    main()
