"""
Is the rank effect real, and does it survive on SHORT lists?

THE TWO OBJECTIONS THIS ANSWERS
  (a) "a claim placed 30th is unlikely to be correct" -- is that measured here, or
      inherited from somewhere? Measure it.
  (b) the effect is described using a 71-item list. What happens when the question
      only admits 4 items, or when an agent emits 500? If the gradient only exists
      on long lists, `rank` is a proxy for VERBOSITY and the v3 ablation
      ("verbosity adds +0.01 over rank") is backwards.

TWO CONFOUNDS, BOTH CONTROLLED
  SUPPORT.  A claim many agents assert is both likelier correct and likelier to sit
  early in every list, so the marginal rank effect conflates rank with support
  count. The solo-only panels remove it: exactly one agent asserts, so rank is the
  only thing varying.

  LENGTH.  A claim from a 3-item list is not comparable to one from a 500-item
  list. Stratifying by the asserting agent's own list length asks whether a rank
  gradient exists WITHIN a length band -- i.e. whether rank is positional at all,
  or merely a restatement of how much that agent wrote.

UNIT
  One row per (claim, asserting agent), not per claim: the question is about where
  an individual agent placed something in its own output.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.rank_effect
"""
import json, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.weighting_v3 import load_rich
from open_ended_aggregation.paths import RESULTS

BANDS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 10 ** 9, "101+")]
RBUCK = [(0, 0, "1st"), (1, 2, "2-3"), (3, 6, "4-7"), (7, 14, "8-15"),
         (15, 29, "16-30"), (30, 69, "31-70"), (70, 10 ** 9, "71+")]


def band(v, table):
    for lo, hi, lbl in table:
        if lo <= v <= hi:
            return lbl
    return table[-1][2]


def rate(d):
    h, t = d
    return (100.0 * h / t) if t else float("nan")


def main():
    print("=" * 94)
    print("  RANK EFFECT -- measured, and stratified by how long the asserting "
          "agent's list was")
    print("=" * 94)

    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()

    rows = []                      # (rank, list_len, support_count, correct)
    for q in qids:
        for ms, gi, rk, key in cand[q]:
            ok = gi is not None
            for m, r in rk.items():
                rows.append((r, nlist[q][m], len(ms), ok))
    print(f"\n  {len(qids)} questions, {len(models)} agents, "
          f"{len(rows):,} (claim, asserting agent) pairs")
    solo = [r for r in rows if r[2] == 1]
    print(f"  {len(solo):,} of them are solo assertions "
          f"({100*len(solo)/len(rows):.1f}%)\n")

    # ---------------------------------------------------------------- 1. pooled
    t = collections.defaultdict(lambda: [0, 0])
    for r, L, c, ok in rows:
        b = band(r, RBUCK); t[b][0] += ok; t[b][1] += 1
    print("  [1] ALL assertions, by absolute position in the asserting agent's list")
    print(f"      {'position':>10}{'n':>12}{'P(correct)':>13}")
    for _, _, lbl in RBUCK:
        if t[lbl][1]:
            print(f"      {lbl:>10}{t[lbl][1]:>12,}{rate(t[lbl]):>12.1f}%")

    # ------------------------------------------- 2. solo only, rank x list length
    print(f"\n  [2] SOLO assertions only (no support confound), "
          f"absolute position x list length")
    g = collections.defaultdict(lambda: [0, 0])
    for r, L, c, ok in solo:
        g[(band(L, BANDS), band(r, RBUCK))][0] += ok
        g[(band(L, BANDS), band(r, RBUCK))][1] += 1
    hdr = "".join(f"{lbl:>11}" for _, _, lbl in RBUCK)
    print(f"      {'list len':>10}{hdr}")
    for _, _, Lb in BANDS:
        cells = []
        for _, _, rb in RBUCK:
            d = g[(Lb, rb)]
            cells.append(f"{rate(d):>10.1f}%" if d[1] >= 100 else f"{'-':>11}")
        print(f"      {Lb:>10}" + "".join(cells))
    print("      (cells with fewer than 100 observations are suppressed)")

    # -------------------------------- 3. solo only, RELATIVE rank x list length
    print(f"\n  [3] SOLO assertions only, RELATIVE position (rank / list length) "
          f"x list length")
    q5 = [(0.0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60%"),
          (0.6, 0.8, "60-80%"), (0.8, 1.01, "80-100%")]
    g2 = collections.defaultdict(lambda: [0, 0])
    for r, L, c, ok in solo:
        rel = r / max(1, L)
        rb = next(lbl for lo, hi, lbl in q5 if lo <= rel < hi)
        g2[(band(L, BANDS), rb)][0] += ok
        g2[(band(L, BANDS), rb)][1] += 1
    hdr = "".join(f"{lbl:>11}" for _, _, lbl in q5)
    print(f"      {'list len':>10}{hdr}")
    for _, _, Lb in BANDS:
        cells = []
        for _, _, rb in q5:
            d = g2[(Lb, rb)]
            cells.append(f"{rate(d):>10.1f}%" if d[1] >= 100 else f"{'-':>11}")
        print(f"      {Lb:>10}" + "".join(cells))

    # ------------------------------------------------- 4. is it length, or rank?
    print(f"\n  [4] DISENTANGLING -- solo assertions, first position only, by "
          f"list length")
    print(f"      if the gradient in [2] were really a length effect, this row "
          f"would fall too")
    g3 = collections.defaultdict(lambda: [0, 0])
    for r, L, c, ok in solo:
        if r == 0:
            g3[band(L, BANDS)][0] += ok; g3[band(L, BANDS)][1] += 1
    print(f"      {'list len':>10}{'n':>12}{'P(correct | 1st)':>19}")
    for _, _, Lb in BANDS:
        if g3[Lb][1]:
            print(f"      {Lb:>10}{g3[Lb][1]:>12,}{rate(g3[Lb]):>18.1f}%")

    # ------------------------------------------------------ 5. within-band slope
    print(f"\n  [5] WITHIN-BAND SLOPE -- Spearman(rank, correct) computed inside "
          f"each length band")
    print(f"      a negative value means later positions are worse, holding "
          f"list length roughly fixed")
    print(f"      {'list len':>10}{'n':>12}{'rho':>10}{'1st vs last-quintile':>24}")
    summary = {}
    for _, _, Lb in BANDS:
        sub = [(r, ok) for r, L, c, ok in solo if band(L, BANDS) == Lb]
        if len(sub) < 200:
            continue
        x = np.array([s[0] for s in sub], dtype=float)
        yv = np.array([float(s[1]) for s in sub])
        xr = x.argsort().argsort().astype(float)
        rho = float(np.corrcoef(xr, yv)[0, 1])
        top = g2[(Lb, "0-20%")]; bot = g2[(Lb, "80-100%")]
        cmp_ = (f"{rate(top):.1f}% vs {rate(bot):.1f}%"
                if top[1] >= 100 and bot[1] >= 100 else "-")
        summary[Lb] = dict(n=len(sub), rho=rho, first=rate(top), last=rate(bot))
        print(f"      {Lb:>10}{len(sub):>12,}{rho:>10.3f}{cmp_:>24}")

    json.dump(summary, open(RESULTS / "rank_effect.json", "w"), indent=2)
    print("\n  wrote results/rank_effect.json")


if __name__ == "__main__":
    main()
