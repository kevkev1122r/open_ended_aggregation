"""
Every agent subset, through the SAME script as the main table (weighting_v5,
log-scale answer count, boosted arms off), so the full-pool row of the sweep is
the main-table number by construction (a reviewer caught the old sweep and main
table disagreeing on 5-agent CWQ).

Sizes: 3 .. pool size (QAMPARI 3..6 of 8, as in the paper; CWQ/WebQSP 3..5 of 5).
Per subset and arm: mean F1, gain over counting, paired bootstrap 95% CI.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.subset_sweep_v5 --bench cwq [--workers 4]
"""
import argparse, itertools, json, os, subprocess, sys, statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def bootstrap(a, b, n=10000, seed=0):
    """Paired percentile 95% CI on mean(a) - mean(b) over questions (x100), vectorised."""
    d = np.asarray(a, float) - np.asarray(b, float)
    reps = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.quantile(reps, .025), 100 * np.quantile(reps, .975)
from open_ended_aggregation.paths import RESULTS, ROOT

POOLS = {
    "qampari": ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5", "MAI-Thinking-1",
                "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "grok-4.3"],
    "cwq": ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5", "MAI-Thinking-1", "grok-4.3"],
    "webqsp": ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5", "MAI-Thinking-1", "grok-4.3"],
}
SIZES = {"qampari": (3, 4, 5, 6), "cwq": (3, 4, 5), "webqsp": (3, 4, 5)}
# one BLAS thread per subset process: N parallel subsets x 15 numpy threads each
# oversubscribed the machine (load average ~250)
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
ARMS = ["weighted", "pattern", "LEAN (logistic)", "count + budget", "weighted + budget",
        "pattern + budget", "LEAN + budget"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(POOLS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    outdir = RESULTS / "sweeps" / a.bench
    outdir.mkdir(parents=True, exist_ok=True)
    subsets = [c for k in SIZES[a.bench] for c in itertools.combinations(POOLS[a.bench], k)]
    py = str(ROOT / "venv" / "bin" / "python")

    def run(sub):
        out = outdir / ("__".join(m.split("-")[0] + m[-3:] for m in sub) + ".json")
        if not out.exists():          # resumable
            subprocess.run([py, "-m", "open_ended_aggregation.analysis.weighting_v5", "--bench", a.bench,
                            "--no-gbm", "--pool", ",".join(sub), "--out", str(out)],
                           cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV1)
        return sub, out

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sub, out) in enumerate(ex.map(run, subsets), 1):
            if not out.exists():
                print(f"  FAILED {sub}", flush=True)
                continue
            d = json.load(open(out))
            pq = d["per_question"]
            row = {"k": len(sub), "agents": list(sub), "nq": d["n_questions"], "count": d["f1"]["count"],
                   "best": max(d["singles"].values()), "best_single": d["best_single"]}
            for arm in ARMS:
                dlt, lo, hi = bootstrap(pq[arm], pq["count"])
                row[arm] = d["f1"][arm]
                row[arm + " vs count"] = dlt
                row[arm + " sig"] = bool(lo > 0)
            dlt, lo, hi = bootstrap(pq["count"], d["best_single_per_question"])
            row["count vs best"] = dlt
            rows.append(row)
            if i % 10 == 0 or i == len(subsets):
                print(f"  {i}/{len(subsets)} subsets", flush=True)
    json.dump(rows, open(RESULTS / f"subset_sweep_v5_{a.bench}.json", "w"), indent=1)

    print(f"\n  {a.bench}: gain over counting, mean per size (significant / subsets)")
    hdr = "  size  n   best   count  " + "  ".join(f"{x[:16]:>16}" for x in ARMS)
    print(hdr)
    for k in SIZES[a.bench]:
        r = [x for x in rows if x["k"] == k]
        if not r:
            continue
        cells = "  ".join(f"{statistics.mean(x[arm + ' vs count'] for x in r):+6.2f} ({sum(x[arm + ' sig'] for x in r):>3}/{len(r):<3})"
                          for arm in ARMS)
        print(f"  {k:>4} {len(r):>3} {statistics.mean(x['best'] for x in r):6.2f} "
              f"{statistics.mean(x['count'] for x in r):6.2f}  {cells}")


if __name__ == "__main__":
    main()
