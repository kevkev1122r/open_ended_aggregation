"""
Plan action 6: judge-audit the QAMPARI pilot outputs, to measure how much of
each arm's "wrong" output is actually right but missing from the gold list.

Reads the n80 pilot's saved outputs (results/synth_prompts_qampari_n80.jsonl,
results/synth_cache_qampari.jsonl); one judge call per question covers every
output item of every arm (and the scorer's keep set) that does not match gold,
de-duplicated. No arm is re-run.

Reported per arm: exact-match precision (as in the pilot), judge-corrected
precision = (gold matches + judged correct) / output items, and corrected F1
with recall unchanged (gold recall cannot be corrected without a complete gold).

SPEND  paid.PaidCaller: cached, capped, --confirm only.

Usage:
  ./venv/bin/python -m open_ended_aggregation.analysis.judge_audit            # estimate
  ./venv/bin/python -m open_ended_aggregation.analysis.judge_audit --confirm  # spends
"""
import argparse, json, statistics

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.benchmarks.qampari import norm, parse_list, score_set
from open_ended_aggregation.methods.judge import (JUDGE_MODEL, list_judge_prompt,
                                                  list_judge_labels)
from open_ended_aggregation.paid import PaidCaller, NotConfirmed, CapReached
from open_ended_aggregation.paths import RESULTS

ARMS = ("text", "p", "pscramble", "solo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n80")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    side = [json.loads(l) for l in open(RESULTS / f"synth_prompts_qampari_{a.tag}.jsonl")]
    cache = {}
    for l in open(RESULTS / "synth_cache_qampari.jsonl"):
        o = json.loads(l)
        cache[o["key"]] = o
    items = {it["qid"]: it for it in QA.load_items(100000)}

    jobs = {}
    for s in side:
        q = s["qid"]
        gold = [{norm(x) for x in [g["answer_text"]] + list(g.get("aliases") or []) if str(x).strip()}
                for g in items[q]["gold"]]
        outs = {arm: parse_list((cache.get(s["prompts"][arm]["key"]) or {}).get("text", ""))
                for arm in ARMS if arm in s["prompts"]}
        outs["scorer"] = list(s["scorer_keep"])
        unmatched = sorted({norm(x) for v in outs.values() for x in v
                            if norm(x) and not any(norm(x) in g for g in gold)})
        jobs[q] = (outs, gold, unmatched)

    caller = PaidCaller(RESULTS / "judge_cache_qampari.jsonl", a.judge, a.effort,
                        cap=a.max_cost, confirm=a.confirm, json_mode=True)
    prompts = {q: list_judge_prompt(items[q]["question"],
                                    [g["answer_text"] for g in items[q]["gold"]], un)
               for q, (_, _, un) in jobs.items() if un}
    caller.estimate(list(prompts.values()), f"list judge ({a.judge})")
    if not a.confirm:
        print("  No --confirm: nothing was sent.")
        return

    per = {arm: {"p_exact": [], "p_judged": [], "f1_exact": [], "f1_judged": []}
           for arm in list(ARMS) + ["scorer"]}
    rescued = total_un = unparsed = 0
    try:
        for q, (outs, gold, un) in jobs.items():
            ok_extra = set()
            if un:
                txt, _ = caller(prompts[q], tag=q)
                lab = list_judge_labels(txt, len(un))
                unparsed += len(un) - len(lab)
                ok_extra = {un[i] for i, l in lab.items() if l == "correct"}
                rescued += len(ok_extra)
                total_un += len(un)
            for arm, pred in outs.items():
                if arm not in per:
                    continue
                keys = [norm(x) for x in pred if norm(x)]
                pe, rc, fe = score_set(pred, items[q]["gold"])
                if keys:
                    good = sum(1 for k in keys if any(k in g for g in gold) or k in ok_extra)
                    pj = good / len(keys)
                else:
                    pj = 0.0
                fj = 2 * pj * rc / (pj + rc) if pj + rc else 0.0
                per[arm]["p_exact"].append(pe); per[arm]["p_judged"].append(pj)
                per[arm]["f1_exact"].append(fe); per[arm]["f1_judged"].append(fj)
    except (NotConfirmed, CapReached) as e:
        print(f"  STOPPED: {e}")

    print(f"\n  judge {a.judge}: {rescued}/{total_un} gold-unmatched items judged correct "
          f"({100*rescued/max(1,total_un):.1f}%); {unparsed} unlabelled")
    print(f"  {'arm':10s}{'P exact':>9}{'P judged':>10}{'F1 exact':>10}{'F1 judged':>11}")
    for arm, d in per.items():
        if d["p_exact"]:
            print(f"  {arm:10s}{100*statistics.mean(d['p_exact']):9.2f}"
                  f"{100*statistics.mean(d['p_judged']):10.2f}"
                  f"{100*statistics.mean(d['f1_exact']):10.2f}{100*statistics.mean(d['f1_judged']):11.2f}")
    json.dump({"judge": a.judge, "rescued": rescued, "unmatched": total_un, "per_arm": per},
              open(RESULTS / f"judge_audit_qampari_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
