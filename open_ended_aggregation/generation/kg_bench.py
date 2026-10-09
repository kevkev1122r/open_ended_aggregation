"""Generation for CWQ and WebQSP, filtered to questions with >10 gold answers.

Mirrors benchmarks/qampari.py: same prompt, same 12,000-token budget, same
temperature-0 single sample per agent, so the generations are comparable to the
QAMPARI pool. The agent pool is five, not eight -- the gpt-5.4 family has no
deployment on the current endpoint (checked 29 Aug 2026).

Usage:  ./venv/bin/python -m open_ended_aggregation.generation.kg_bench cwq
"""
import sys, os, json, time, collections

from open_ended_aggregation.paths import ROOT as _ROOT
from open_ended_aggregation.benchmarks import qampari as Q

HERE = str(_ROOT)
POOL = ["grok-4.3", "Kimi-K2.5", "MAI-Thinking-1",
        "DeepSeek-V4-Flash", "Cohere-command-a-plus-05-2026"]
MAX_PASSES = 12


def load_items(name):
    return json.load(open(f"{HERE}/data/{name}_items.json"))


def coverage(items, path):
    have = collections.defaultdict(set)
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); have[r["qid"]].add(r["model"])
            except Exception:
                pass
    got = sum(len(have[it["qid"]] & set(POOL)) for it in items)
    full = sum(1 for it in items if len(have[it["qid"]] & set(POOL)) == len(POOL))
    return got, len(items) * len(POOL), full


if __name__ == "__main__":
    from open_ended_aggregation.backends import azure as AZ
    name = sys.argv[1]
    items = load_items(name)
    out = f"{HERE}/data/{name}_gen.jsonl"
    Q.POOL, Q.GEN_PATH = POOL, out
    t0 = time.time()
    print("=" * 70, flush=True)
    print(f"  {name.upper()}  n={len(items)}  agents={len(POOL)}  -> {os.path.basename(out)}", flush=True)
    print("=" * 70, flush=True)
    prev = -1
    for p in range(1, MAX_PASSES + 1):
        got, want, full = coverage(items, out)
        if got >= want:
            print(f"  pass {p}: COMPLETE ({got}/{want})", flush=True); break
        if got == prev:
            print(f"  pass {p}: no progress, stopping at {got}/{want}", flush=True); break
        prev, before = got, got
        try:
            Q.generate(items)
        except Exception as e:
            print(f"  pass {p} raised {type(e).__name__}: {e}", flush=True)
        got, want, full = coverage(items, out)
        print(f"  pass {p}: +{got-before} -> {got}/{want} ({100*got/want:.1f}%), "
              f"{full} full q, throttled={AZ.usage['throttled']}, ${AZ.spend():.2f}", flush=True)
    got, want, full = coverage(items, out)
    print(f"\n  FINAL {got}/{want} cells, {full}/{len(items)} complete questions, "
          f"${AZ.spend():.2f}, {(time.time()-t0)/60:.0f} min", flush=True)
