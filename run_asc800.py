"""
Generation for the paper-ASC-with-agents run: 800 QAMPARI questions x 8 agents.

Seeds data/qampari_asc800.jsonl from the two existing generation files (they
already cover 4,944 of the 6,400 cells) then backfills the rest in passes, since
a long-lived generate() silently drops ~50% of calls under sustained load while a
fresh one does not -- measured 2026-08-18, see run_imbalanced.py.

Usage:  ./venv/bin/python run_asc800.py [n]
"""
import sys, os, json, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

POOL = ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5",
        "MAI-Thinking-1", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "grok-4.3"]
OUT = f"{HERE}/data/qampari_asc800.jsonl"
SRC = [f"{HERE}/data/qampari_gen.jsonl", f"{HERE}/data/qampari_imb.jsonl"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
MAX_PASSES = 15


def seed(qids):
    have = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l); have.add((r["qid"], r["model"]))
            except Exception:
                pass
    n = 0
    with open(OUT, "a") as f:
        for p in SRC:
            if not os.path.exists(p):
                continue
            for l in open(p):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                k = (r["qid"], r["model"])
                if r["qid"] in qids and r["model"] in POOL and k not in have:
                    have.add(k); f.write(json.dumps(r) + "\n"); n += 1
    print(f"  seeded {n} cached rows -> {os.path.basename(OUT)}", flush=True)


def coverage(items):
    have = collections.defaultdict(set)
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l); have[r["qid"]].add(r["model"])
            except Exception:
                pass
    got = sum(len(have[it["qid"]] & set(POOL)) for it in items)
    full = sum(1 for it in items if len(have[it["qid"]] & set(POOL)) == len(POOL))
    return got, len(items) * len(POOL), full


if __name__ == "__main__":
    import run_qampari as Q, azure_backend as AZ
    t0 = time.time()
    items = Q.load_items(N)
    qids = {it["qid"] for it in items}
    print("=" * 74)
    print(f"  PAPER-ASC GENERATION   n={len(items)}  agents={len(POOL)}")
    print("=" * 74, flush=True)
    seed(qids)

    Q.POOL = POOL
    Q.GEN_PATH = OUT
    prev = -1
    for p in range(1, MAX_PASSES + 1):
        got, want, full = coverage(items)
        if got >= want:
            print(f"  pass {p}: COMPLETE ({got}/{want}, {full} full questions)", flush=True)
            break
        if got == prev:
            print(f"  pass {p}: no progress -- stopping at {got}/{want} "
                  f"({full} full questions)", flush=True)
            break
        prev, before = got, got
        try:
            Q.generate(items)
        except Exception as e:
            print(f"  pass {p} raised {type(e).__name__}: {e}", flush=True)
        got, want, full = coverage(items)
        print(f"  pass {p}: +{got-before} -> {got}/{want} ({100*got/want:.1f}%), "
              f"{full} full questions, throttled={AZ.usage['throttled']}", flush=True)

    got, want, full = coverage(items)
    print(f"\n  FINAL {got}/{want} cells ({100*got/want:.1f}%), "
          f"{full}/{len(items)} complete questions, {(time.time()-t0)/60:.0f} min", flush=True)
