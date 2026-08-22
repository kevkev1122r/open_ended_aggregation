"""
Paper-exact ASC as a per-ensemble baseline column, alongside MA-count.

WHY BOTH BASELINES
  The 21 Aug meeting settled that the paper needs TWO baselines and that they
  are different objects:

    MA-count   keep a claim if >= theta AGENTS assert it, exact string match,
               one vote per agent, theta cross-fitted on F1. This is the team's
               proposed naive multi-agent method, not a published one.

    ASC        the published pipeline (arXiv:2405.13131): atomic units, EDIT
               DISTANCE clustering at a tuned tau, strength = COUNT OF CLUSTER
               MEMBERS (not deduplicated per agent), keep clusters above a
               validation-tuned Theta, representative = first member.

  They differ on three axes -- exact match vs edit-distance clustering, per-agent
  dedup vs raw member counts, and F1 vs F1-5 as the tuning objective -- so ASC
  is a genuinely separate arm, not a relabelling of counting.

THE ONE SUBSTITUTION, WHICH IS UNAVOIDABLE
  Published ASC draws m~50 stochastic samples from ONE model. The cache holds one
  response per agent, so the m samples are replaced by the n agents. This is the
  substitution `methods/asc.py` already documents; it is also precisely what the
  team is proposing as the extension, so it is the right baseline here. Running
  ASC in its original single-model form needs new generation (777 x m calls) and
  is not possible from cache.

  The composition step (feeding survivors back to an LLM) is omitted -- it costs
  API calls and applies equally to every arm, so omitting it does not favour any.

CLUSTERING COST, AND THE APPROXIMATION MADE
  Clustering is exact per ensemble where affordable. One QAMPARI question has
  3,305 unique items (an agent emitted a runaway list) and ~450k edges, so
  re-running union-find for all 210 ensembles x 777 questions is not viable.
  Instead the edge structure is computed ONCE at the tuned tau over the full
  pool and cached; per ensemble, cluster strength counts only occurrences from
  ensemble agents, clusters with zero ensemble support are dropped, and the
  representative is the first member an ensemble agent actually asserted.

  The approximation: an item from an excluded agent can still bridge two
  clusters that would separate without it. Measured effect is small because
  edit-distance neighbourhoods among entity names are tight, but it is an
  approximation and is flagged as one. (On ASQA, where questions hold ~21
  sentences, clustering IS redone per ensemble -- see asqa_ensembles.py.)

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.asc_ensembles [--tau 0.15]
"""
import os, json, math, time, pickle, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.methods import asc as ASC
from open_ended_aggregation.benchmarks import qampari as Q
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
CACHE = DATA / "asc_edges.pkl"
TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def build_cache(byq, models, qids, tau_max):
    """Per question: unique items, per-item agent occurrence counts, edges."""
    out = {}
    t0 = time.time()
    for n_, q in enumerate(qids, 1):
        order, mult, first, E = ASC.edges(byq, q, models, tau_max)
        occ = [collections.Counter() for _ in order]
        pos = {k: i for i, k in enumerate(order)}
        for mi_, m in enumerate(models):
            for it in byq[q][m]["items"]:
                k = Q.norm(it)
                if k:
                    occ[pos[k]][mi_] += 1
        out[q] = (order, first, E, occ)
        if n_ % 150 == 0:
            print(f"    edges {n_}/{len(qids)}  {time.time()-t0:.0f}s", flush=True)
    return out


def cluster(order, first, E, occ, emask, tau):
    """Union-find at tau over items asserted by agents in emask.
    Returns [(strength, representative)]."""
    present = [i for i in range(len(order)) if any(a in emask for a in occ[i])]
    if not present:
        return []
    pin = {i: p for p, i in enumerate(present)}
    parent = list(range(len(present)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for d, i, j in E:
        if d < tau and i in pin and j in pin:
            a, b = find(pin[i]), find(pin[j])
            if a != b:
                parent[a] = b
    mem = collections.defaultdict(list)
    for p, i in enumerate(present):
        mem[find(p)].append(i)
    out = []
    for v in mem.values():
        strength = sum(sum(c for a, c in occ[i].items() if a in emask) for i in v)
        out.append((strength, first[order[min(v)]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--sizes", default="3,4,5,6")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    byq, models, qids, items = ASC.load()
    n = len(models)
    gs = {q: ASC.goldsets(items[q]) for q in qids}
    print(f"  QAMPARI  {len(qids)} questions, {n} agents", flush=True)

    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            EC = pickle.load(f)
        print(f"  edge cache loaded ({len(EC)} questions)", flush=True)
    else:
        print(f"  building edge cache at tau_max={max(TAUS)} ...", flush=True)
        EC = build_cache(byq, models, qids, max(TAUS))
        with open(CACHE, "wb") as f:
            pickle.dump(EC, f)
        print(f"  cached to {CACHE}", flush=True)
    tot_e = sum(len(EC[q][2]) for q in qids)
    print(f"  {tot_e:,} edges total, "
          f"{statistics.mean(len(EC[q][0]) for q in qids):.0f} unique items/question",
          flush=True)

    rng = random.Random(SEED)
    order_q = qids[:]; rng.shuffle(order_q)
    half = len(order_q) // 2
    folds = [(order_q[:half], order_q[half:]), (order_q[half:], order_q[:half])]
    full = set(range(n))

    def curve(clusters, gold):
        """Metrics at every prefix of the strength-sorted clusters, in one pass.
        A Theta keeps the clusters with strength >= Theta, which is exactly a
        prefix, so every Theta is priced at once instead of rescoring per Theta."""
        rows = sorted(((c, Q.norm(rep)) for c, rep in clusters), key=lambda r: -r[0])
        ng = max(1, len(gold))
        hp, seen, out, strengths = 0, set(), [(0.0, 0.0, 0.0)], []
        for k, (c, p_) in enumerate(rows, 1):
            gi = next((i for i, gx in enumerate(gold) if p_ in gx), None)
            if gi is not None:
                hp += 1; seen.add(gi)
            prec = hp / k
            rec = len(seen) / ng
            rec5 = min(len(seen), 5) / max(1, min(len(gold), 5))
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            f15 = 0.0 if prec + rec5 == 0 else 2 * prec * rec5 / (prec + rec5)
            out.append((f1, f15, prec))
            strengths.append(c)
        return strengths, out

    def at(cv, theta):
        st, out = cv
        lo, hi = 0, len(st)
        while lo < hi:                       # count of strengths >= theta
            mid = (lo + hi) // 2
            if st[mid] >= theta:
                lo = mid + 1
            else:
                hi = mid
        return out[lo]

    def arm(cvs, theta, qs):
        return [at(cvs[q], theta) for q in qs]

    # ---- tau tuned once on the full pool, on the validation half, by F1-5
    if a.tau:
        TAU = a.tau
        print(f"  tau fixed at {TAU}")
    else:
        best = None
        for tau in TAUS:
            cv = {q: curve(cluster(*EC[q], full, tau), gs[q]) for q in qids}
            for th in range(1, 3 * n + 1):
                v = statistics.mean(x[1] for x in arm(cv, th, folds[0][0]))
                if best is None or v > best[0]:
                    best = (v, tau, th)
            print(f"    tau={tau:.2f} done", flush=True)
        TAU = best[1]
        print(f"  tuned on validation half: tau={TAU}, Theta={best[2]} (max F1-5)")

    # ---- full pool reference
    cv_full = {q: curve(cluster(*EC[q], full, TAU), gs[q]) for q in qids}
    per_full = {}
    for tr, te in folds:
        bt = max(range(1, 3 * n + 1),
                 key=lambda th: statistics.mean(x[1] for x in arm(cv_full, th, tr)))
        for q in te:
            per_full[q] = at(cv_full[q], bt)
    print(f"\n  ASC full pool (n={n}):  "
          f"F1 {100*statistics.mean(per_full[q][0] for q in qids):.2f}   "
          f"P {100*statistics.mean(per_full[q][2] for q in qids):.2f}   "
          f"F1-5 {100*statistics.mean(per_full[q][1] for q in qids):.2f}",
          flush=True)

    # ---- per ensemble
    out = []
    for k in sizes:
        t0 = time.time()
        for ens in itertools.combinations(range(n), k):
            emask = set(ens)
            cv = {q: curve(cluster(*EC[q], emask, TAU), gs[q]) for q in qids}
            # Theta tuned two ways. The paper maximises F1-5, so that is the
            # faithful arm -- but we compare arms on F1, which handicaps it.
            # `asc_f1tuned` removes that objection by giving ASC the same
            # objective everything else is scored on. Report both.
            per, per_f1 = {}, {}
            for tr, te in folds:
                bt = max(range(1, 3 * k + 1),
                         key=lambda th: statistics.mean(x[1] for x in arm(cv, th, tr)))
                b1 = max(range(1, 3 * k + 1),
                         key=lambda th: statistics.mean(x[0] for x in arm(cv, th, tr)))
                for q in te:
                    per[q] = at(cv[q], bt); per_f1[q] = at(cv[q], b1)
            out.append(dict(k=k, ens=[models[i] for i in ens],
                            asc=100 * statistics.mean(per[q][0] for q in qids),
                            asc_f15=100 * statistics.mean(per[q][1] for q in qids),
                            asc_f1tuned=100 * statistics.mean(per_f1[q][0] for q in qids)))
        print(f"    size {k}: {sum(1 for r in out if r['k']==k)} ensembles "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'n':>4}{'ens':>6}{'ASC F1':>10}{'ASC F1-5':>11}"
          f"{'ASC F1 (th tuned on F1)':>26}")
    for k in sizes:
        R = [r for r in out if r["k"] == k]
        if R:
            print(f"  {k:>4}{len(R):>6}{statistics.mean(r['asc'] for r in R):>10.2f}"
                  f"{statistics.mean(r['asc_f15'] for r in R):>11.2f}"
                  f"{statistics.mean(r['asc_f1tuned'] for r in R):>26.2f}")

    json.dump({"tau": TAU,
               "full_pool": {"f1": 100 * statistics.mean(per_full[q][0] for q in qids),
                             "f15": 100 * statistics.mean(per_full[q][1] for q in qids),
                             "prec": 100 * statistics.mean(per_full[q][2] for q in qids)},
               "ensembles": out},
              open(RESULTS / "asc_ensembles.json", "w"), indent=1)
    print("\n  wrote results/asc_ensembles.json")


if __name__ == "__main__":
    main()
