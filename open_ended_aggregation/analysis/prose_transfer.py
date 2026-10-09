"""
Plan action 5 (long-form), free once results/claims_asqa.json exists:
does the prose scorer predict correct facts for agents it never saw, and which
evidence carries it?

  - all 20 directed 3|3 splits of the 6 ASQA agents, 2-fold over questions, so
    source and target share no agent and no question
  - AUC of the transferred p against the partial label (value matches an ASQA
    short answer), per arm:
        full                 all 15 features
        - position           sentence position removed (the with/without test)
        - contradiction      contradiction columns removed
        contra = silence     contradicting agents treated as silent (the
                             "contradiction != omission" test)
        support only         |S|/k alone, the counting baseline
  - bootstrap over splits is not needed: every arm is scored on the same claims,
    so wins are counted per split.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.prose_transfer
"""
import itertools, json, statistics, sys

import numpy as np

from open_ended_aggregation.analysis.claim_extract import load_asqa
from open_ended_aggregation.methods.prose_features import (ProseScorer, featurize, auc,
                                                            auc_within, split_questions)
from open_ended_aggregation.paths import RESULTS

ARMS = {"full": {}, "- position": {"drop": ("position",)},
        "- contradiction": {"drop": ("contradiction",)},
        "contra = silence": {"merge_contra": True}}


def main():
    path = RESULTS / "claims_asqa.json"
    if not path.exists():
        sys.exit("  results/claims_asqa.json not found: run analysis/claim_extract first.")
    claims = json.load(open(path))
    _, items, pool, _ = load_asqa(None)
    halves = split_questions(claims)
    res = {k: [] for k in list(ARMS) + ["support only"]}
    rwq = {k: [] for k in res}
    n_claims = []
    for A in itertools.combinations(pool, 3):
        B = [m for m in pool if m not in A]
        for src, tgt in ((list(A), B), (B, list(A))):
            for tr, te in (halves, halves[::-1]):
                Xf, y, rq, _ = featurize(claims, te, tgt, items)
                if not len(y) or y.min() == y.max():
                    continue
                n_claims.append(len(y))
                for arm, kw in ARMS.items():
                    try:
                        sc = ProseScorer(claims, tr, src, items, **kw)
                    except ValueError:
                        continue
                    X = Xf if not kw.get("merge_contra") else featurize(
                        claims, te, tgt, items, merge_contra=True)[0]
                    res[arm].append(auc(sc.predict(X), y))
                    rwq[arm].append(auc_within(sc.predict(X), y, rq))
                res["support only"].append(auc(Xf[:, 1], y))
                rwq["support only"].append(auc_within(Xf[:, 1], y, rq))
    if not n_claims:
        sys.exit("  no scorable splits (too few labelled claims)")
    print(f"\n  ASQA prose scorer, zero-label transfer across agent pools: "
          f"{len(res['full'])} (split, fold) cases, median {int(np.median(n_claims))} target claims")
    print(f"    {'':18s} {'within-question':>22s}   {'pooled':>22s}")
    for arm in res:
        cells = []
        for d in (rwq, res):
            v, base = d[arm], d["full"]
            wins = sum(1 for a, b in zip(base, v) if a > b)
            cells.append(f"{statistics.mean(v):.3f}" + (f" (full wins {wins:2d}/{len(v)})"
                                                         if arm != "full" else " " * 20))
        print(f"    {arm:18s} {cells[0]:>22s}   {cells[1]:>22s}")
    json.dump({"pooled": res, "within_question": rwq},
              open(RESULTS / "prose_transfer.json", "w"), indent=1)
    print("  wrote results/prose_transfer.json")


if __name__ == "__main__":
    main()
