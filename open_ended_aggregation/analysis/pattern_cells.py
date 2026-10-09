"""
Support-pattern cell occupancy, and the spread the ceiling is made of.

WHY THIS EXISTS
  Section 3.1 argues that any rule reading only S(c) partitions candidates into
  at most 2^n cells and applies one decision per cell. Support counting is the
  coarsening of that partition onto |S(c)|. So everything reliability weighting
  can add over counting lives in the SPREAD of P(correct | S=s) across the cells
  that share a support count. This module measures that spread directly.

  It also answers the self-review's request for cell occupancy on both
  benchmarks, which is the quantity that turns "ASQA is estimating on thin data"
  from an assertion into something a reader can check.

  Rates here are the pooled empirical rates over all claims, not cross-fitted.
  They describe the partition; the method's own arms remain cross-fitted.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.pattern_cells
Output: results/pattern_cells.json
"""
import collections
import json

from open_ended_aggregation.analysis.weighting_v3 import load_rich
from open_ended_aggregation.paths import RESULTS


def cells_for_qampari():
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    n = len(models)
    hit = collections.Counter()
    tot = collections.Counter()
    for q in qids:
        for ms, gi, rk, key in cand[q]:
            s = frozenset(ms)
            tot[s] += 1
            if gi is not None:
                hit[s] += 1
    return n, tot, hit


def summarise(n, tot, hit):
    claims = sum(tot.values())
    correct = sum(hit.values())
    cells = []
    for s, t in tot.items():
        cells.append({"k": len(s), "n": t, "rate": 100.0 * hit[s] / t})
    cells.sort(key=lambda c: (c["k"], -c["n"]))

    by_k = collections.defaultdict(lambda: [0, 0])
    for s, t in tot.items():
        by_k[len(s)][0] += t
        by_k[len(s)][1] += hit[s]

    per_k = []
    for k in sorted(by_k):
        t, h = by_k[k]
        occ = [c for c in cells if c["k"] == k]
        # only cells with enough claims to have a meaningful rate
        solid = [c for c in occ if c["n"] >= 30]
        rates = [c["rate"] for c in solid]
        per_k.append({
            "k": k,
            "claims": t,
            "rate": 100.0 * h / t,
            "cells_occupied": len(occ),
            "cells_possible": _choose(n, k),
            "cells_ge30": len(solid),
            "spread_ge30": (max(rates) - min(rates)) if len(rates) > 1 else None,
        })
    return {
        "agents": n,
        "claims": claims,
        "base_rate": 100.0 * correct / claims,
        "cells_occupied": len(tot),
        "cells_possible": 2 ** n,
        "median_claims_per_occupied_cell": sorted(tot.values())[len(tot) // 2],
        "per_support_count": per_k,
        "cells": cells,
    }


def _choose(n, k):
    from math import comb
    return comb(n, k)


def main():
    n, tot, hit = cells_for_qampari()
    out = summarise(n, tot, hit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pattern_cells.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"  {out['claims']:,} claims, base rate {out['base_rate']:.2f}%")
    print(f"  {out['cells_occupied']} of {out['cells_possible']} support "
          f"patterns occupied; median {out['median_claims_per_occupied_cell']} "
          f"claims per occupied cell\n")
    print("   |S|    claims   rate    occupied/possible   cells>=30   "
          "rate spread across those cells")
    for r in out["per_support_count"]:
        sp = f"{r['spread_ge30']:.1f} pts" if r["spread_ge30"] is not None else "--"
        print(f"   {r['k']:3d}  {r['claims']:9,}  {r['rate']:5.2f}%   "
              f"{r['cells_occupied']:4d}/{r['cells_possible']:<4d}   "
              f"{r['cells_ge30']:6d}      {sp}")
    print("\n  wrote results/pattern_cells.json")


if __name__ == "__main__":
    main()
