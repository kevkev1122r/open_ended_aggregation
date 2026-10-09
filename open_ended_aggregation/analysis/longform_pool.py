"""
Pool the long-form results of every (block, fold) that exists into one paired
report. Free: reads results/longform_asqa*_fold*.json only.

Each question appears in exactly one (block, fold), where its scorer was fitted
on the source agents over the OTHER half of its block -- so pooling keeps the
zero-label design. Blocks: A = the 50-question pilot, B = the next 140.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.longform_pool
"""
import json

from open_ended_aggregation.analysis.longform_rungs import report
from open_ended_aggregation.paths import RESULTS


def main():
    files = sorted(RESULTS.glob("longform_asqa_fold*.json")) + \
        sorted(RESULTS.glob("longform_asqa_block*_fold*.json"))
    per, arms, spend = {}, None, 0.0
    for f in files:
        d = json.load(open(f))
        dup = set(per) & set(d["per_question"])
        assert not dup, f"{f.name}: {len(dup)} questions already pooled"
        per.update(d["per_question"])
        arms = arms or d["arms"]
        assert d["arms"] == arms, f"{f.name}: different arms"
        spend += d.get("spend", 0) + d.get("judge_spend", 0)
        print(f"  {f.name}: {len(d['per_question'])} questions, scorer AUC "
              f"{d['scorer']['auc_scorer']:.3f} vs support {d['scorer']['auc_support']:.3f}")
    report(per, arms, judged=True)
    json.dump({"arms": arms, "files": [f.name for f in files], "per_question": per},
              open(RESULTS / "longform_asqa_pooled.json", "w"), indent=1)


if __name__ == "__main__":
    main()
