"""
ASQA merging arms scored under DR, not STR-EM.

WHY THIS EXISTS
  analysis/merge_asqa.py evaluates the clustered merging arm with STR-EM, which
  is recall-only: concatenating every agent maximises it by construction, so
  every arm collapsed to union and we wrongly concluded ASQA could not adjudicate
  a filter.

  ASQA's headline metric is DR = geometric mean(ROUGE-L, Disambig-F1)
  (Stelmakh et al., EMNLP 2022). ROUGE-L against a ~2-sentence human reference
  punishes a 454-word concatenation, so under DR the union is NOT free. Measured:
  union DR 24.68 vs best single agent 29.01. That is the tradeoff a filter needs.

  DR* here uses STR-EM in place of Disambig-F1, on the authors' statement that
  the two measure the same aspect. Labelled a proxy throughout; a faithful DR
  needs the RoBERTa-SQuADv2 reader.

ATOMIC UNIT
  A SENTENCE, not an entity. Different agents never write character-identical
  sentences, so an exact-match count filter keeps nothing (measured: theta=2
  retains ~1 word). Clustering is therefore semantic -- mpnet embeddings,
  single-linkage at cosine >= TAU -- exactly as in merge_asqa.py.

SCHEMES  (thresholds cross-fitted; nothing tuned on the questions it scores)
  count | marginal | solo | nb-full | pattern    -- same family as
  analysis/weighting_schemes.py, so QAMPARI/QUEST/ASQA are directly comparable.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.asqa_merge_dr
"""
import sys, json, math, re, random, collections, statistics, itertools

import numpy as np
import pandas as pd

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

TAU = 0.85
SEED = 0
FOLDS = 5
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(t):
    return [s.strip() for s in _SENT.split(str(t)) if len(s.strip().split()) >= 3]


def main():
    df = pd.read_parquet(DATA / "asqa_dev.parquet")
    refs = {f"asqa:{i:04d}": [a["long_answer"] for a in r["annotations"] if a.get("long_answer")]
            for i, r in df.iterrows()}
    items = {it["qid"]: it for it in A.load_items(100000)}
    A._ITEMS.update(items)

    recs = [json.loads(l) for l in open(DATA / "asqa_imb.jsonl")]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    POOL = sorted({r["model"] for r in recs})
    qids = [q for q, d in byq.items() if len(d) == len(POOL) and q in items and refs.get(q)]
    print(f"  ASQA  {len(qids)} questions, {len(POOL)} agents, DR* scoring")

    # ---- embed every sentence once
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    per_q, flat = {}, []
    for q in qids:
        rec = [(m, s) for m in POOL for s in sentences(byq[q][m]["resp"])]
        per_q[q] = rec
        flat += [s for _, s in rec]
    print(f"  {len(flat):,} sentences ({len(flat)/len(qids):.1f}/question), embedding...",
          flush=True)
    E = enc.encode(flat, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True).astype(np.float64)
    off, i = {}, 0
    for q in qids:
        off[q] = (i, i + len(per_q[q])); i += len(per_q[q])

    # ---- cluster once per question -> [(frozenset(models), representative)]
    clusters = {}
    for q in qids:
        rec = per_q[q]; a, b = off[q]; V = E[a:b]; n = len(rec)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        if n:
            S = np.clip(V @ V.T, -1, 1)
            for x in range(n):
                for y in range(x + 1, n):
                    if S[x, y] >= TAU:
                        parent[find(x)] = find(y)
        mem, mods = collections.defaultdict(list), collections.defaultdict(set)
        for idx, (m, s) in enumerate(rec):
            r0 = find(idx); mem[r0].append(s); mods[r0].add(m)
        clusters[q] = [(frozenset(mods[r0]), max(mem[r0], key=len)) for r0 in mem]
    print(f"  mean clusters/question {statistics.mean(len(clusters[q]) for q in qids):.1f}",
          flush=True)

    # ---- scoring, cached on the kept-set (ROUGE-L is the expensive part)
    cache = {}

    def dr_of(q, kept_idx):
        key = (q, kept_idx)
        if key in cache:
            return cache[key]
        txt = " ".join(clusters[q][i][1] for i in kept_idx)
        se = A.str_em(txt, items[q]["short_sets"])
        rl = rouge_l(txt, refs[q])
        v = (math.sqrt(max(0.0, se) * max(0.0, rl)), se, rl)
        cache[key] = v
        return v

    # ---- a "correct" cluster = one whose text covers >=1 interpretation.
    # Needed only to FIT the weighting schemes, never to score them.
    def cluster_correct(q, rep):
        return A.str_em(rep, items[q]["short_sets"]) > 0

    rows = {q: [(ms, cluster_correct(q, rep)) for ms, rep in clusters[q]] for q in qids}

    def est(train):
        allr = [r for q in train for r in rows[q]]
        pos = sum(1 for _, c in allr if c); neg = len(allr) - pos
        n = collections.Counter(); c_ = collections.Counter()
        sn = collections.Counter(); sc = collections.Counter()
        tp = collections.Counter(); fp = collections.Counter()
        bypat = collections.defaultdict(lambda: [0, 0]); bycnt = collections.defaultdict(lambda: [0, 0])
        for ms, ok in allr:
            for m in ms:
                n[m] += 1; c_[m] += ok
                (tp if ok else fp)[m] += 1
            if len(ms) == 1:
                m = next(iter(ms)); sn[m] += 1; sc[m] += ok
            bypat[ms][0] += ok; bypat[ms][1] += 1
            bycnt[len(ms)][0] += ok; bycnt[len(ms)][1] += 1
        marg = {m: (c_[m] + 1) / (n[m] + 2) for m in POOL}
        solo = {m: (sc[m] + 1) / (sn[m] + 2) for m in POOL}
        nb = {m: (math.log(((tp[m] + 1) / (pos + 2)) / ((fp[m] + 1) / (neg + 2))),
                  math.log((1 - (tp[m] + 1) / (pos + 2)) / (1 - (fp[m] + 1) / (neg + 2))))
              for m in POOL}
        prior = math.log((pos + 1) / (neg + 1))
        cr = {k: (v[0] + 1) / (v[1] + 2) for k, v in bycnt.items()}
        pat = {p: (h + 20.0 * cr.get(len(p), .3)) / (t + 20.0) for p, (h, t) in bypat.items()}
        # UNIFORM control: equal weights, but a CONTINUOUS threshold. If this
        # matches the reliability-weighted arms, the gain is threshold
        # granularity, not reliability -- the same control that proved the
        # QAMPARI gain was not fractional-threshold artefact.
        return {
            "count":    lambda ms: float(len(ms)),
            "uniform":  lambda ms: 0.25 * len(ms),
            "marginal": lambda ms: sum(marg[m] for m in ms),
            "solo":     lambda ms: sum(solo[m] for m in ms),
            "nb-full":  lambda ms: prior + sum(nb[m][0] if m in ms else nb[m][1] for m in POOL),
            "pattern":  lambda ms: pat.get(ms, cr.get(len(ms), .3)),
        }

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    out = collections.defaultdict(dict)
    for f in range(FOLDS):
        tr = [q for q in qids if fold[q] != f]
        te = [q for q in qids if fold[q] == f]
        sc = est(tr)
        for arm, fn in sc.items():
            ths = sorted({fn(frozenset(c)) for r in range(len(POOL) + 1)
                          for c in itertools.combinations(POOL, r)})
            best = None
            for t in ths:
                v = statistics.mean(
                    dr_of(q, tuple(i for i, (ms, _) in enumerate(clusters[q])
                                   if fn(ms) >= t - 1e-12))[0] for q in tr)
                if best is None or v > best[0]:
                    best = (v, t)
            th = best[1]
            for q in te:
                out[arm][q] = dr_of(q, tuple(i for i, (ms, _) in enumerate(clusters[q])
                                             if fn(ms) >= th - 1e-12))

    singles = {m: [dr_of(q, tuple(i for i, (ms, _) in enumerate(clusters[q]) if m in ms))
                   for q in qids] for m in POOL}
    bm = max(POOL, key=lambda m: statistics.mean(x[0] for x in singles[m]))
    best = [x[0] for x in singles[bm]]
    union = [dr_of(q, tuple(range(len(clusters[q]))))[0] for q in qids]

    print(f"\n  {'arm':<20}{'DR*':>8}{'STR-EM':>9}{'ROUGE-L':>9}"
          f"{'vs count':>21}{'vs best single':>21}")
    print("  " + "-" * 88)

    def line(lbl, vals, ref=None):
        dr = [v[0] for v in vals]; se = [v[1] for v in vals]; rl = [v[2] for v in vals]
        cells = ""
        if ref is not None:
            d, l, h = bootstrap(dr, ref)
            cells = f"{d:+7.2f}[{l:+6.2f},{h:+6.2f}]{'*' if (l>0 or h<0) else ' '}"
        d2, l2, h2 = bootstrap(dr, best)
        print(f"  {lbl:<20}{100*statistics.mean(dr):8.2f}{100*statistics.mean(se):9.2f}"
              f"{100*statistics.mean(rl):9.2f}{cells:>21}"
              f"{d2:+7.2f}[{l2:+6.2f},{h2:+6.2f}]{'*' if (l2>0 or h2<0) else ' '}")

    line(f"best single({bm[:9]})", singles[bm])
    line("union (no filter)", [dr_of(q, tuple(range(len(clusters[q])))) for q in qids])
    cnt = [out["count"][q][0] for q in qids]
    for arm in ["count", "uniform", "marginal", "solo", "nb-full", "pattern"]:
        line(arm, [out[arm][q] for q in qids], None if arm == "count" else cnt)
    print("\n  DR* = sqrt(ROUGE-L x STR-EM), proxy for published DR")
    json.dump({a: 100*statistics.mean(out[a][q][0] for q in qids)
               for a in out}, open(RESULTS / "asqa_merge_dr.json", "w"), indent=2)


if __name__ == "__main__":
    main()
