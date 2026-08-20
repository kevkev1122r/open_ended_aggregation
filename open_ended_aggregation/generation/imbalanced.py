"""
QAMPARI + ASQA on an IMBALANCED pool, with BACKFILL PASSES until convergence.

WHY PASSES AND NOT ONE LONG RUN
  Measured 2026-08-18: a single long-lived generate() drops ~50% of calls under
  sustained load. A FRESH process succeeds 36/36 on the identical prompts, budget
  and worker count -- so the loss is a property of the long-lived process, not of
  the models, the questions, or throttling (usage["throttled"] stayed 0).

  generate() writes a row only `if txt:`, so every dropped call vanishes with no
  trace: the (qid, model) cell is simply absent, and the progress line prints only
  spend and `empty`, neither of which moves. That is the v1 silent-failure mode
  wearing a third hat.

  Dropped cells ARE recoverable -- done_keys() skips what is already written, so
  re-running backfills exactly the gaps. This driver therefore runs generate()
  repeatedly, in a fresh-ish state each pass, until a pass adds nothing. It prints
  rows added and cells still missing after every pass, so the loss is VISIBLE and
  convergence is checkable instead of assumed.

POOL
  The pool we designed (Ministral-3B, gpt-4o, ...) is not deployed and deploying
  needs the `az` CLI, absent here. This is the widest spread among models that
  already answer, verified by direct probe.

    gpt-5.4-nano < gpt-5.4-mini < gpt-5.4 < DeepSeek-V4-Flash < MAI-Thinking-1 < grok-4.3

  Capability of the three gpt-5.4 size variants is UNMEASURED -- establishing
  whether they form a real ladder is the point of this run.

  gpt-5.5 / gpt-5.6-sol excluded despite being live: they reject temperature=0,
  and run_v2.py:72 warns that one model at temperature 1 beside peers at 0
  invalidates the comparison.

Usage:  ./venv/bin/python run_imbalanced.py [n] [max_passes]
"""
import sys, os, json, time, collections

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
POOL = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
        "DeepSeek-V4-Flash", "MAI-Thinking-1", "grok-4.3"]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
MAX_PASSES = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def coverage(path, items):
    have = collections.defaultdict(set)
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); have[r["qid"]].add(r["model"])
            except Exception:
                pass
    want = len(items) * len(POOL)
    got = sum(len(have[it["qid"]] & set(POOL)) for it in items)
    full = sum(1 for it in items if len(have[it["qid"]] & set(POOL)) == len(POOL))
    return got, want, full


def drive(label, mod, gen_path, items):
    import azure_backend as AZ
    mod.POOL = POOL
    mod.GEN_PATH = gen_path
    print("\n" + "=" * 74)
    print(f"  {label}  n={len(items)}  pool={len(POOL)}  ->  {os.path.basename(gen_path)}")
    print("=" * 74, flush=True)
    prev = -1
    for p in range(1, MAX_PASSES + 1):
        got, want, full = coverage(gen_path, items)
        if got >= want:
            print(f"  pass {p}: COMPLETE ({got}/{want} cells, {full} full questions)", flush=True)
            break
        if got == prev:
            print(f"  pass {p}: no progress last pass -- stopping at {got}/{want} "
                  f"({full} full questions)", flush=True)
            break
        prev = got
        t0 = time.time()
        before = got
        try:
            mod.generate(items)
        except Exception as e:
            print(f"  pass {p} raised {type(e).__name__}: {e}", flush=True)
        got, want, full = coverage(gen_path, items)
        print(f"  pass {p}: +{got-before} cells -> {got}/{want} "
              f"({100*got/want:.1f}%), {full} full questions, "
              f"{time.time()-t0:.0f}s, throttled={AZ.usage['throttled']}", flush=True)
    got, want, full = coverage(gen_path, items)
    print(f"  FINAL {label}: {got}/{want} cells ({100*got/want:.1f}%), "
          f"{full}/{len(items)} complete questions", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(f"pool: {POOL}", flush=True)

    import run_qampari as Q
    drive("QAMPARI", Q, f"{HERE}/data/qampari_imb.jsonl", Q.load_items(N))

    import run_asqa as A
    a_items = A.load_items(N)
    A._ITEMS.update({it["qid"]: it for it in a_items})
    drive("ASQA", A, f"{HERE}/data/asqa_imb.jsonl", a_items)

    import azure_backend as AZ
    print(f"\nTOTAL wall {(time.time()-t0)/3600:.2f} h   "
          f"priced spend ${AZ.spend():.3f}", flush=True)
    print(f"NOTE: {sorted(AZ.usage['unpriced'])} have no PRICE entry, "
          f"so spend UNDERSTATES actual cost.", flush=True)
