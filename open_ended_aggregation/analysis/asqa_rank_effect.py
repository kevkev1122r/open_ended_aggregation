"""
The rank-effect diagnostic from analysis/rank_effect.py, ported to ASQA.

WHY
  On QAMPARI the position gradient is real but LENGTH-CONDITIONAL: measured on
  solo assertions, Spearman(rank, correct) is +0.084 on 1-5 item lists (nothing),
  -0.102 on 21-100, -0.069 on 101+, and 70% of QAMPARI's solo assertions sit in
  lists of 101+ items. The feature works there because the agents emit long
  enumerations.

  ASQA responses average ~3.5 sentences. That is squarely inside the band where
  QAMPARI shows NO usable gradient. So this script tests a prediction made in
  advance: the confidence-signal slot should be close to empty on ASQA, which
  would explain why `ours` loses to reliability weighting there
  (asqa_ours_ensembles.py) rather than leaving it as an unexplained defeat.

WHAT MAPS ONTO WHAT
    QAMPARI                        ASQA
    claim (normalised entity)      sentence cluster
    rank in the agent's list       sentence index in the agent's response
    agent's list length            number of sentences the agent wrote
    claim is in gold               representative covers an interpretation

  Same two controls as the QAMPARI version: solo-only panels remove the support
  confound, length bands remove the verbosity confound.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.asqa_rank_effect
"""
import json, collections

import numpy as np

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.analysis.asqa_ensembles import build, cluster
from open_ended_aggregation.paths import RESULTS

# ASQA responses are short, so the bands are an order of magnitude tighter
# than QAMPARI's -- that is the whole point of the comparison.
BANDS = [(1, 2, "1-2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 10 ** 9, "8+")]
RBUCK = [(0, 0, "1st"), (1, 1, "2nd"), (2, 2, "3rd"), (3, 4, "4-5"),
         (5, 10 ** 9, "6+")]


def band(v, table):
    for lo, hi, lbl in table:
        if lo <= v <= hi:
            return lbl
    return table[-1][2]


def rate(d):
    h, t = d
    return (100.0 * h / t) if t else float("nan")


def main():
    print("=" * 92)
    print("  ASQA RANK EFFECT -- sentence position inside the agent's own response")
    print("=" * 92)

    POOL, qids, per_q, sim, items, refs = build()
    E = set(POOL)

    rows = []                       # (position, n_sentences, support, correct)
    nsent_all = []
    for q in qids:
        c = collections.Counter(); posn = []
        for m, s in per_q[q]:
            posn.append(c[m]); c[m] += 1
        nsent = {m: max(1, c[m]) for m in POOL}
        nsent_all.extend(nsent.values())
        for ms, rep, gs in cluster(per_q[q], sim[q], E):
            ok = A.str_em(rep, items[q]["short_sets"]) > 0
            first = {}
            for g in gs:
                m = per_q[q][g][0]
                first[m] = min(first.get(m, 10 ** 6), posn[g])
            for m, p in first.items():
                rows.append((p, nsent[m], len(ms), bool(ok)))

    solo = [r for r in rows if r[2] == 1]
    print(f"\n  {len(qids)} questions, {len(POOL)} agents, "
          f"{len(rows):,} (cluster, asserting agent) pairs")
    print(f"  {len(solo):,} solo ({100*len(solo)/len(rows):.1f}%)   "
          f"mean sentences per response {np.mean(nsent_all):.1f}, "
          f"median {int(np.median(nsent_all))}, max {max(nsent_all)}")
    d = collections.Counter(band(L, BANDS) for _, L, _, _ in rows)
    tot = sum(d.values())
    print("  response-length distribution:  " +
          "   ".join(f"{lbl} {100*d[lbl]/tot:.0f}%" for _, _, lbl in BANDS))

    # -------------------------------------------------------------- 1. pooled
    t = collections.defaultdict(lambda: [0, 0])
    for p, L, c, ok in rows:
        b = band(p, RBUCK); t[b][0] += ok; t[b][1] += 1
    print("\n  [1] ALL assertions, by sentence position in the agent's response")
    print(f"      {'position':>10}{'n':>10}{'P(covers)':>13}")
    for _, _, lbl in RBUCK:
        if t[lbl][1]:
            print(f"      {lbl:>10}{t[lbl][1]:>10,}{rate(t[lbl]):>12.1f}%")

    # ------------------------------------- 2. solo only, position x length band
    print("\n  [2] SOLO only (no support confound), position x response length")
    g = collections.defaultdict(lambda: [0, 0])
    for p, L, c, ok in solo:
        k = (band(L, BANDS), band(p, RBUCK))
        g[k][0] += ok; g[k][1] += 1
    print(f"      {'n sent':>10}" + "".join(f"{lbl:>11}" for _, _, lbl in RBUCK))
    for _, _, Lb in BANDS:
        cells = []
        for _, _, rb in RBUCK:
            dd = g[(Lb, rb)]
            cells.append(f"{rate(dd):>10.1f}%" if dd[1] >= 60 else f"{'-':>11}")
        print(f"      {Lb:>10}" + "".join(cells))
    print("      (cells with fewer than 60 observations are suppressed)")

    # --------------------------------------------- 3. within-band Spearman
    print("\n  [3] WITHIN-BAND SLOPE -- Spearman(position, covers), solo only")
    print(f"      QAMPARI for reference:  1-5 items +0.084 | 6-20 -0.048 | "
          f"21-100 -0.102 | 101+ -0.069")
    print(f"      {'n sent':>10}{'n':>10}{'rho':>10}{'1st':>10}{'last':>10}")
    summary = {}
    for _, _, Lb in BANDS:
        sub = [(p, ok) for p, L, c, ok in solo if band(L, BANDS) == Lb]
        if len(sub) < 150:
            continue
        x = np.array([s[0] for s in sub], dtype=float)
        yv = np.array([float(s[1]) for s in sub])
        if yv.std() == 0:
            continue
        xr = x.argsort().argsort().astype(float)
        rho = float(np.corrcoef(xr, yv)[0, 1])
        lo = [s for s in sub if s[0] == 0]
        hi = [s for s in sub if s[0] >= 2]
        f = 100 * sum(s[1] for s in lo) / max(1, len(lo))
        l = 100 * sum(s[1] for s in hi) / max(1, len(hi))
        summary[Lb] = dict(n=len(sub), rho=rho, first=f, later=l)
        print(f"      {Lb:>10}{len(sub):>10,}{rho:>10.3f}{f:>9.1f}%{l:>9.1f}%")

    # -------------------------- 4. pooled slope, the number to quote alongside
    x = np.array([r[0] for r in solo], dtype=float)
    yv = np.array([float(r[3]) for r in solo])
    xr = x.argsort().argsort().astype(float)
    pooled = float(np.corrcoef(xr, yv)[0, 1])
    print(f"\n  [4] pooled over all lengths, solo only:  rho = {pooled:+.3f}  "
          f"(QAMPARI equivalent -0.121)")
    summary["_pooled"] = pooled

    json.dump(summary, open(RESULTS / "asqa_rank_effect.json", "w"), indent=2)
    print("\n  wrote results/asqa_rank_effect.json")


if __name__ == "__main__":
    main()
