"""
Every ensemble of size 3..6 drawn from the 6 ASQA agents, scored under DR*.

WHY THIS IS THE EXPERIMENT THAT MATTERS
  On QAMPARI the gain of reliability weighting over counting decays monotonically
  with ensemble size -- +2.25 at n=3, +1.90 at n=4, +0.40 at n=5, -0.36 at n=6 --
  and the decay survives oracle weights, so it is not estimation noise. The
  reading is that with enough models the support COUNT becomes a sufficient
  statistic and model identity stops carrying information.

  That is one dataset. ASQA is the test of whether it is a property of
  aggregation or a property of QAMPARI, and it is a genuinely different setting:

    atomic unit    a SENTENCE cluster, not an entity string
    agreement      semantic (mpnet cosine >= 0.85), not exact match
    metric         DR* = sqrt(ROUGE-L x STR-EM), which has a length penalty
                   built in, rather than set F1

  If the crossover replicates across all three of those changes, it is the
  paper's central figure. If it does not, the QAMPARI curve is a QAMPARI fact
  and must be reported as one.

CLUSTERING IS REDONE PER ENSEMBLE
  Clustering once on all 6 agents and then intersecting would leak: a sentence
  from an excluded agent can bridge two clusters under single linkage, and the
  representative could be a sentence no ensemble member ever wrote. The cosine
  matrix is computed once per question; the union-find is rerun over the
  ensemble's sentences only, and the representative is always drawn from inside
  the ensemble.

HONEST THRESHOLDS
  2-fold cross-fitted, both directions pooled -- the same protocol as
  analysis/ensemble_sizes.py, so the two datasets' curves are directly
  comparable. Weights are fitted on the training half too.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_ensembles [--sizes 3,4]
"""
import os, re, json, math, random, argparse, itertools, collections, statistics

import numpy as np
import pandas as pd

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

TAU = 0.85
SEED = 0
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(t):
    return [s.strip() for s in _SENT.split(str(t)) if len(s.strip().split()) >= 3]


def build():
    """Per question: the sentences, who wrote each, the cosine matrix, refs."""
    df = pd.read_parquet(DATA / "asqa_dev.parquet")
    refs = {f"asqa:{i:04d}": [a["long_answer"] for a in r["annotations"]
                              if a.get("long_answer")] for i, r in df.iterrows()}
    items = {it["qid"]: it for it in A.load_items(100000)}
    A._ITEMS.update(items)

    recs = [json.loads(l) for l in open(DATA / "asqa_imb.jsonl")]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    POOL = sorted({r["model"] for r in recs})
    qids = sorted(q for q, d in byq.items()
                  if len(d) == len(POOL) and q in items and refs.get(q))

    per_q = {q: [(m, s) for m in POOL for s in sentences(byq[q][m]["resp"])]
             for q in qids}
    flat = [s for q in qids for _, s in per_q[q]]
    print(f"  ASQA  {len(qids)} questions, {len(POOL)} agents, "
          f"{len(flat):,} sentences", flush=True)

    cache = DATA / "asqa_sent_emb.npy"
    if os.path.exists(cache):
        E = np.load(cache)
        assert len(E) == len(flat), "embedding cache stale — delete it"
        print("  embeddings loaded from cache", flush=True)
    else:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        E = enc.encode(flat, batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True).astype(np.float32)
        np.save(cache, E)
        print("  embeddings computed and cached", flush=True)

    sim, off, i = {}, {}, 0
    for q in qids:
        k = len(per_q[q])
        V = E[i:i + k].astype(np.float64)
        sim[q] = np.clip(V @ V.T, -1, 1) >= TAU
        off[q] = i
        i += k
    return POOL, qids, per_q, sim, items, refs


def cluster(per_q_rows, adj, keep):
    """Union-find over the sentences written by agents in `keep`.
    Returns [(frozenset(models), representative_sentence, (sentence ids))]."""
    idx = [i for i, (m, _) in enumerate(per_q_rows) if m in keep]
    pos = {g: j for j, g in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if adj[idx[a], idx[b]]:
                parent[find(a)] = find(b)
    mem = collections.defaultdict(list)
    for j, g in enumerate(idx):
        mem[find(j)].append(g)
    out = []
    for r, gs in mem.items():
        ms = frozenset(per_q_rows[g][0] for g in gs)
        rep = max((per_q_rows[g][1] for g in gs), key=len)
        out.append((ms, rep, tuple(sorted(gs))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3,4,5,6")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    POOL, qids, per_q, sim, items, refs = build()
    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]

    dr_cache = {}

    def dr(q, sids):
        """DR* of the concatenation of the given sentences. ROUGE-L dominates
        the cost, so this is memoised on the exact sentence set -- which repeats
        heavily both across thresholds and across ensembles."""
        key = (q, sids)
        v = dr_cache.get(key)
        if v is None:
            txt = " ".join(s for s in sids_text(q, sids))
            se = A.str_em(txt, items[q]["short_sets"])
            rl = rouge_l(txt, refs[q])
            v = dr_cache[key] = math.sqrt(max(0.0, se) * max(0.0, rl))
        return v

    def sids_text(q, sids):
        return [per_q[q][i][1] for i in sids]

    out = []
    for k in sizes:
        for ens in itertools.combinations(POOL, k):
            E = set(ens)
            cl = {q: cluster(per_q[q], sim[q], E) for q in qids}
            # the reps a filter can keep, indexed by support pattern
            pats = [frozenset(c) for r in range(1, k + 1)
                    for c in itertools.combinations(ens, r)]

            def kept_sids(q, keep):
                s = []
                for ms, rep, gs in cl[q]:
                    if ms in keep:
                        s.append(gs[0] if len(gs) == 1 else
                                 max(gs, key=lambda g: len(per_q[q][g][1])))
                return tuple(sorted(s))

            corr = {q: [(ms, A.str_em(rep, items[q]["short_sets"]) > 0)
                        for ms, rep, _ in cl[q]] for q in qids}

            scored = {"count": {}, "weighted": {}}
            for train, test in folds:
                n = collections.Counter(); c = collections.Counter()
                for q in train:
                    for ms, ok in corr[q]:
                        for m in ms:
                            n[m] += 1; c[m] += ok
                w = {m: (c[m] + 1) / (n[m] + 2) for m in ens}
                rules = {"count": lambda p: float(len(p)),
                         "weighted": lambda p, _w=w: sum(_w[m] for m in p)}
                for arm, fn in rules.items():
                    best = None
                    for t in sorted({fn(p) for p in pats}):
                        keep = {p for p in pats if fn(p) >= t - 1e-12}
                        v = statistics.mean(dr(q, kept_sids(q, keep)) for q in train)
                        if best is None or v > best[0]:
                            best = (v, keep)
                    for q in test:
                        scored[arm][q] = dr(q, kept_sids(q, best[1]))

            singles = {m: [dr(q, kept_sids(q, {p for p in pats if m in p}))
                           for q in qids] for m in ens}
            bm = max(ens, key=lambda m: statistics.mean(singles[m]))
            best_v = singles[bm]
            cnt = [scored["count"][q] for q in qids]
            wgt = [scored["weighted"][q] for q in qids]
            d1, l1, h1 = bootstrap(cnt, best_v)
            d2, l2, h2 = bootstrap(wgt, best_v)
            d3, l3, h3 = bootstrap(wgt, cnt)
            pr = sorted((statistics.mean(singles[m]) for m in ens), reverse=True)
            out.append(dict(k=k, ens=list(ens), best_model=bm,
                            best=100 * statistics.mean(best_v),
                            count=100 * statistics.mean(cnt),
                            weighted=100 * statistics.mean(wgt),
                            cnt_vs_best=d1, cnt_sig=(l1 > 0 or h1 < 0),
                            wgt_vs_best=d2, wgt_sig=(l2 > 0 or h2 < 0),
                            wgt_vs_cnt=d3, wgt_cnt_sig=(l3 > 0 or h3 < 0),
                            dominance=pr[0] / max(1e-9, statistics.mean(pr[1:]))))
        print(f"    size {k}: {sum(1 for r in out if r['k']==k)} ensembles done "
              f"({len(dr_cache):,} DR evals cached)", flush=True)

    print(f"\n  ASQA, DR*  --  mean over ensembles by size")
    print(f"  {'size':>5}{'n':>5}{'best single':>13}{'count':>9}{'weighted':>10}"
          f"{'cnt-best':>11}{'wgt-best':>11}{'wgt-cnt':>10}")
    print("  " + "-" * 76)
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        if not R:
            continue
        print(f"  {k:>5}{len(R):>5}{statistics.mean(r['best'] for r in R):>13.2f}"
              f"{statistics.mean(r['count'] for r in R):>9.2f}"
              f"{statistics.mean(r['weighted'] for r in R):>10.2f}"
              f"{statistics.mean(r['cnt_vs_best'] for r in R):>+11.2f}"
              f"{statistics.mean(r['wgt_vs_best'] for r in R):>+11.2f}"
              f"{statistics.mean(r['wgt_vs_cnt'] for r in R):>+10.2f}")

    print(f"\n  how often each arm SIGNIFICANTLY beats the alternative")
    print(f"  {'size':>5}{'count > best':>15}{'weighted > best':>17}{'weighted > count':>18}")
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        if not R:
            continue
        x = sum(1 for r in R if r["cnt_sig"] and r["cnt_vs_best"] > 0)
        y = sum(1 for r in R if r["wgt_sig"] and r["wgt_vs_best"] > 0)
        z = sum(1 for r in R if r["wgt_cnt_sig"] and r["wgt_vs_cnt"] > 0)
        print(f"  {k:>5}{x:>10}/{len(R):<4}{y:>12}/{len(R):<4}{z:>13}/{len(R):<4}")

    if len(out) > 2:
        xs = [r["dominance"] for r in out]; ys = [r["wgt_vs_cnt"] for r in out]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        if den > 0:
            print(f"\n  corr(relative dominance, weighted-count) over {len(out)} "
                  f"ensembles = {num/den:+.3f}")

    json.dump(out, open(RESULTS / "asqa_ensembles.json", "w"), indent=1)
    print("  wrote results/asqa_ensembles.json")


if __name__ == "__main__":
    main()
