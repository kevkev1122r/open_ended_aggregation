"""
Does OMISSION STRENGTH actually predict correctness, and how strongly?

WHY THIS EXISTS
  analysis/weighting_v3.py's docstring calls omission strength "a large, entirely
  unmodelled signal" and motivates the `omit` feature block with it. That claim
  was never measured -- the block's contribution was only ever seen through an
  ablation of the whole model. This measures the raw effect directly, the same
  way analysis/rank_effect.py does for position.

  Verdict when first run (23 Aug 2026): real, correctly signed, and about HALF
  the strength of rank. Pooled Spearman -0.048 against rank's -0.102 in its best
  band. "Large" is not supported; "real but small" is.

THE TWO CONFOUNDS
  SUPPORT.  Restricted to solo claims -- exactly one agent asserts -- so the
  effect cannot be support count in disguise.

  THE ASSERTER'S OWN VERBOSITY.  A claim from a 2,431-item list is not comparable
  to one from a 3-item list, and the asserter's length correlates with the pool's.
  Stratifying by it asks whether silent-agent volume carries anything ON TOP of
  how much the asserting agent itself wrote.

  ONE CONFOUND THAT SURVIVES, AND MUST BE DISCLOSED
  Questions where every agent writes a lot are plausibly broader questions. So
  silent-agent volume partly proxies question difficulty rather than evidence
  about this particular claim. Nothing here separates those.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.omission_effect
"""
import json, math, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.weighting_v3 import load_rich
from open_ended_aggregation.paths import RESULTS

# asserting agent's own list length
ABANDS = [(1, 20, "1-20"), (21, 100, "21-100"), (101, 10 ** 9, "101+")]
# mean log(list length) over the silent agents; e^1.5 ~ 4 items, e^4.5 ~ 90
SBANDS = [(0.0, 1.5, "<1.5"), (1.5, 2.5, "1.5-2.5"), (2.5, 3.5, "2.5-3.5"),
          (3.5, 4.5, "3.5-4.5"), (4.5, 99.0, ">4.5")]


def band(v, table):
    for lo, hi, lbl in table:
        if lo <= v < hi or (hi >= 10 ** 8 and v >= lo):
            return lbl
    return table[-1][2]


def main():
    print("=" * 84)
    print("  OMISSION STRENGTH -- does what the SILENT agents wrote predict "
          "correctness?")
    print("=" * 84)

    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()

    print(f"\n  [1] list lengths per agent -- the spread the feature relies on")
    print(f"      {'agent':<30}{'median':>8}{'mean':>9}{'max':>8}")
    lens = {}
    for m in models:
        L = [nlist[q][m] for q in qids]
        lens[m] = dict(median=statistics.median(L), mean=statistics.mean(L),
                       max=max(L))
        print(f"      {m.split('/')[-1][:28]:<30}{int(statistics.median(L)):>8}"
              f"{statistics.mean(L):>9.1f}{max(L):>8}")

    rows = []
    for q in qids:
        for ms, gi, rk, key in cand[q]:
            if len(ms) != 1:
                continue
            m = next(iter(ms))
            sil = [x for x in models if x != m]
            rows.append((nlist[q][m],
                         statistics.mean(math.log(nlist[q][x]) for x in sil),
                         gi is not None))

    print(f"\n  [2] P(correct) by silent-agent volume, solo claims only")
    print(f"      stratified by the ASSERTING agent's own list length")
    g = collections.defaultdict(lambda: [0, 0])
    for La, s, ok in rows:
        k = (band(La, ABANDS), band(s, SBANDS))
        g[k][0] += ok; g[k][1] += 1
    print(f"\n      {'asserter len':>14}" + "".join(f"{l:>12}" for _, _, l in SBANDS))
    out = {}
    for _, _, Ab in ABANDS:
        cells = []
        for _, _, Sb in SBANDS:
            h, t = g[(Ab, Sb)]
            cells.append(f"{100*h/t:>11.1f}%" if t >= 150 else f"{'-':>12}")
            if t >= 150:
                out[f"{Ab}|{Sb}"] = dict(n=t, p=100 * h / t)
        print(f"      {Ab:>14}" + "".join(cells))
    print(f"      columns = mean log(list length) of the silent agents")
    print(f"      (cells with fewer than 150 observations suppressed)")

    x = np.array([r[1] for r in rows])
    y = np.array([float(r[2]) for r in rows])
    xr = x.argsort().argsort().astype(float)
    rho = float(np.corrcoef(xr, y)[0, 1])
    print(f"\n  [3] pooled Spearman(silent-agent volume, correct) = {rho:+.3f}"
          f"   n={len(rows):,}")
    print(f"      for scale, rank reaches -0.102 within its best length band "
          f"(see rank_effect.py)")

    json.dump(dict(pooled_rho=rho, n=len(rows), cells=out, lengths=lens),
              open(RESULTS / "omission_effect.json", "w"), indent=2)
    print("\n  wrote results/omission_effect.json")


if __name__ == "__main__":
    main()
