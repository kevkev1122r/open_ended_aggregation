"""
ASQA replication of the QAMPARI merging result — HANDOFF_CURRENT.md §4.1.

DIFFERENCES FROM analyze_merge_asqa.py
  1. Bug A (per-model dedup) was ALREADY correct there ("one vote per model per
     cluster"). Kept as-is. The handoff's claim that this file carries Bug A is
     wrong — see verify_qampari_independent.py for the same finding on QAMPARI.
  2. Bug B is real here: the old sweep was np.linspace(0.15,1.0,10)*wsum, i.e.
     10 points spanning the whole range. Clusters are re-thresholded on a fine
     grid instead. Clustering is done ONCE per question and cached, so a
     200-point sweep plus 20 shuffled derangements costs almost nothing.
  3. Adds the UNIFORM and SHUFFLED controls that made the QAMPARI result
     credible, and reports which model occupies the "trusted solo" slot.

Usage:  ./venv/bin/python analyze_merge_asqa_fixed.py
"""
import os, sys, json, re, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_asqa as R

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
TAU = 0.85
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    return [s.strip() for s in _SENT.split(str(text))
            if len(s.strip().split()) >= 3]


def boot(a, b, n=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    r = np.random.default_rng(SEED)
    s = d[r.integers(0, len(d), (n, len(d)))].mean(axis=1) * 100
    return d.mean() * 100, np.percentile(s, 2.5), np.percentile(s, 97.5)


def show(lbl, a, b):
    m, lo, hi = boot(a, b)
    star = "  *" if not (lo < 0 < hi) else ""
    print(f"    {lbl:<30}{m:+7.2f}   [{lo:+6.2f}, {hi:+6.2f}]{star}")


def main():
    items = {it["qid"]: it for it in R.load_items(100000)}
    R._ITEMS.update(items)
    rows = [json.loads(l) for l in open(f"{HERE}/data/asqa_gen.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        if r["qid"] in items:
            by[r["qid"]][r["model"]] = r
    POOL = R.POOL
    qids = sorted(q for q in by if len(by[q]) == len(POOL))
    print("=" * 74)
    print(f"  ASQA REPLICATION   n={len(qids)} complete questions, {len(POOL)} models")
    print("=" * 74)

    strem = np.array([[by[q][m]["strem"] for m in POOL] for q in qids])
    order = np.argsort(-strem.mean(axis=0))
    print("\n  per-model STR-EM:")
    for i in order:
        print(f"    {POOL[i]:<34}{100*strem[:, i].mean():6.2f}%")
    best_i = int(order[0])
    best_vec = strem[:, best_i]
    print(f"  best single = {POOL[best_i]}  {100*best_vec.mean():.2f}%")
    print(f"  capability spread = "
          f"{100*(strem.mean(0).max()-strem.mean(0).min()):.2f} points")

    # ---------- embed + cluster ONCE per question ----------
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    per_q, flat = {}, []
    for q in qids:
        rec = [(m, s) for m in POOL for s in sentences(by[q][m]["resp"])]
        per_q[q] = rec
        flat += [s for _, s in rec]
    print(f"\n  {len(flat):,} sentences ({len(flat)/max(1,len(qids)):.1f} per question)",
          flush=True)
    E = enc.encode(flat, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    off, i = {}, 0
    for q in qids:
        off[q] = (i, i + len(per_q[q])); i += len(per_q[q])

    # clusters[q] = list of (frozenset_of_models, representative_sentence)
    clusters, dup_hits, tot_sent = {}, 0, 0
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
        mem, mods = collections.defaultdict(list), collections.defaultdict(list)
        for idx, (m, s) in enumerate(rec):
            r0 = find(idx); mem[r0].append(s); mods[r0].append(m)
        for r0 in mem:
            tot_sent += len(mods[r0])
            dup_hits += len(mods[r0]) - len(set(mods[r0]))
        clusters[q] = [(frozenset(mods[r0]), max(mem[r0], key=len)) for r0 in mem]
    print(f"  within-model duplicate sentences inside a cluster: "
          f"{dup_hits} / {tot_sent} ({100*dup_hits/max(1,tot_sent):.2f}%)"
          f"   <- Bug A's raw material on ASQA")

    # ---------- exact evaluation ----------
    # A threshold filter on subset sums can realise at most 2^|POOL| distinct
    # behaviours, because support depends only on WHICH SUBSET of models backs a
    # cluster. So the optimum over theta is found EXACTLY by evaluating at every
    # distinct subset sum -- no grid, and Bug B cannot occur by construction.
    SUBSETS = [frozenset(c) for k in range(len(POOL) + 1)
               for c in __import__("itertools").combinations(POOL, k)]

    _cache = {}

    def evaluate(wmap, theta):
        out = np.empty(len(qids))
        for j, q in enumerate(qids):
            kept = tuple(i for i, (ms, _) in enumerate(clusters[q])
                         if sum(wmap[m] for m in ms) >= theta - 1e-9)
            ck = (q, kept)
            v = _cache.get(ck)
            if v is None:
                txt = " ".join(clusters[q][i][1] for i in kept)
                v = _cache[ck] = R.str_em(txt, items[q]["short_sets"])
            out[j] = v
        return out

    def breakpoints(wmaps):
        """Every theta at which the filter can change, for any of these maps."""
        s = set()
        for wm in wmaps:
            for ss in SUBSETS:
                s.add(round(sum(wm[m] for m in ss), 9))
        return sorted(s)

    def sweep(wmap, grid=None):
        best = None
        for th in (grid if grid is not None else breakpoints([wmap])):
            v = evaluate(wmap, th)
            if best is None or v.mean() > best[1].mean():
                best = (float(th), v)
        return best

    # ---------- arms ----------
    ONE = {m: 1.0 for m in POOL}
    cnt = sweep(ONE, [1, 2, 3, 4, 5])
    print(f"\n  ASC count filter        best theta={cnt[0]:.0f}   "
          f"STR-EM {100*cnt[1].mean():6.2f}%")
    for t in [1, 2, 3, 4, 5]:
        print(f"      theta={t}  {100*evaluate(ONE, t).mean():6.2f}%")

    # cross-fitted non-negative weights = mean STR-EM out of fold
    fold = np.random.default_rng(SEED).integers(0, 5, len(qids))
    qi = {q: k for k, q in enumerate(qids)}
    w_cv = {k: strem[fold != k].mean(axis=0) for k in range(5)}
    glob = strem.mean(axis=0)
    print("\n  weights (global):  " +
          "  ".join(f"{POOL[i][:12]}={glob[i]:.3f}" for i in order))
    solo = np.sort(glob)[::-1]
    pairs = sorted(glob[a] + glob[b] for a in range(len(POOL))
                   for b in range(a + 1, len(POOL)))
    print(f"  2nd solo {solo[1]:.4f}   weakest pair {pairs[0]:.4f}"
          f"   -> weighting differs from counting only for theta in "
          f"({solo[1]:.4f}, {pairs[0]:.4f}]")

    fold_maps = [{m: float(w_cv[k][POOL.index(m)]) for m in POOL} for k in range(5)]
    grid = breakpoints(fold_maps)

    def evaluate_cf(theta):
        out = np.empty(len(qids))
        for j, q in enumerate(qids):
            wm = fold_maps[fold[qi[q]]]
            kept = tuple(i for i, (ms, _) in enumerate(clusters[q])
                         if sum(wm[m] for m in ms) >= theta - 1e-9)
            ck = (q, kept)
            v = _cache.get(ck)
            if v is None:
                txt = " ".join(clusters[q][i][1] for i in kept)
                v = _cache[ck] = R.str_em(txt, items[q]["short_sets"])
            out[j] = v
        return out

    best_w, curve = None, []
    for th in grid:
        v = evaluate_cf(th)
        curve.append((float(th), 100 * v.mean()))
        if best_w is None or v.mean() > best_w[1].mean():
            best_w = (float(th), v)
    print(f"\n  WEIGHTED filter (EXACT, {len(grid)} subset-sum breakpoints)   "
          f"best theta={best_w[0]:.4f}   STR-EM {100*best_w[1].mean():6.2f}%")
    lo_b, hi_b = solo[1] * 0.8, pairs[0] * 1.2
    print("    neighbourhood of the reordering band:")
    for th, sc in curve:
        if lo_b <= th <= hi_b:
            print(f"      theta {th:7.4f}   {sc:6.2f}%"
                  + ("   <- best" if abs(th - best_w[0]) < 1e-9 else ""))

    # ---------- controls ----------
    uni = sweep(ONE)
    print(f"\n  CONTROL uniform    best theta={uni[0]:.3f}   "
          f"STR-EM {100*uni[1].mean():6.2f}%")

    rng = np.random.default_rng(123)
    runs = []
    for _ in range(20):
        while True:
            p = rng.permutation(len(POOL))
            if all(p[i] != i for i in range(len(POOL))):
                break
        wm = {POOL[i]: float(glob[p[i]]) for i in range(len(POOL))}
        runs.append(sweep(wm)[1].mean())
    while True:
        p = np.random.default_rng(123).permutation(len(POOL))
        if all(p[i] != i for i in range(len(POOL))):
            break
    wm = {POOL[i]: float(glob[p[i]]) for i in range(len(POOL))}
    shf = sweep(wm)
    print(f"  CONTROL shuffled   mean over 20 derangements "
          f"{100*np.mean(runs):6.2f}%   (fixed one {100*shf[1].mean():6.2f}%)")

    # ---------- what rule is it? ----------
    th = best_w[0]
    g0 = {m: float(glob[POOL.index(m)]) for m in POOL}
    solo_pass = [m for m in POOL if g0[m] >= th]
    all_pairs = all(g0[a] + g0[b] >= th for i, a in enumerate(POOL) for b in POOL[i+1:])
    print(f"\n  rule at theta*: solo-clearing models = {solo_pass or 'none'};"
          f"  every pair clears = {all_pairs}")

    print("\n  SUMMARY")
    print(f"    best single ({POOL[best_i][:22]:<22}) {100*best_vec.mean():6.2f}%")
    print(f"    ASC count   (theta={cnt[0]:.0f})              {100*cnt[1].mean():6.2f}%")
    print(f"    WEIGHTED    (theta={best_w[0]:.3f})          {100*best_w[1].mean():6.2f}%")
    print(f"    uniform control                    {100*uni[1].mean():6.2f}%")
    print(f"    shuffled control                   {100*shf[1].mean():6.2f}%")

    print("\n  DECISIVE COMPARISONS (paired bootstrap, * = CI excludes 0)")
    show("WEIGHTED - best single", best_w[1], best_vec)
    show("WEIGHTED - ASC count", best_w[1], cnt[1])
    show("WEIGHTED - uniform", best_w[1], uni[1])
    show("WEIGHTED - shuffled", best_w[1], shf[1])
    show("ASC count - best single", cnt[1], best_vec)
    show("uniform - ASC count", uni[1], cnt[1])

    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump({"n": len(qids), "tau": TAU, "best_single_model": POOL[best_i],
               "best_single": 100 * best_vec.mean(),
               "asc_count": {"theta": cnt[0], "strem": 100 * cnt[1].mean()},
               "weighted": {"theta": best_w[0], "strem": 100 * best_w[1].mean()},
               "uniform": 100 * uni[1].mean(),
               "shuffled_mean20": 100 * float(np.mean(runs))},
              open(f"{HERE}/results/asqa_merge_fixed.json", "w"), indent=2)
    print(f"\n  wrote results/asqa_merge_fixed.json")


if __name__ == "__main__":
    main()
