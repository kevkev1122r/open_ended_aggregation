"""
Is the ensemble-size crossover real, or an estimation artefact?

THE FINDING UNDER TEST
  On QAMPARI, weighting beats counting by +2.25 at ensemble size 3 and by -0.36
  at size 6 -- a clean monotone decay. Read naively: weighting matters when the
  ensemble is small and becomes unnecessary as it grows.

THE DANGEROUS ALTERNATIVE
  At size 3 you fit 3 weights; at size 6 you fit 6, from the SAME questions.
  More parameters, noisier estimates. That alone predicts weighting decaying with
  ensemble size, with no underlying effect at all. This project has already had a
  headline flip sign (+0.08 -> -1.25) once thresholds were cross-fitted honestly,
  so the burden of proof sits here.

THE CONTROLS
  weighted-cf      weights and threshold fitted on the training half   (honest)
  weighted-oracle  weights and threshold fitted on ALL data            (no estimation error)
  pattern-cf       full support-pattern lookup, cross-fitted           (ceiling, honest)
  pattern-oracle   full support-pattern lookup, fitted on all data     (ceiling, no error)
  uniform          equal weights, continuous threshold                 (must equal count)

  READING THE RESULT
    If weighted-ORACLE still decays with ensemble size -> the crossover is a real
    property of aggregation, and estimation noise is not the explanation.
    If oracle is flat while cf decays -> the crossover is an artefact and must
    not be the paper's headline.

  uniform is a bug check: score = w * count is monotone in count, so thresholding
  it realises exactly the count filters. Any gap means the harness is wrong.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.crossover_controls
"""
import json, math, itertools, collections, statistics, random

from open_ended_aggregation.analysis.ensemble_sizes import load, group, f1_from
from open_ended_aggregation.paths import RESULTS

SEED = 0


def main():
    POOL, bit, qids, cand, ngold = load()
    print(f"  QAMPARI  {len(qids)} questions, pool of {len(POOL)}")
    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]

    singles = {m: {q: f1_from(group(cand[q], bit[m]), [bit[m]], ngold[q]) for q in qids}
               for m in POOL}

    def fit_weights(groups, ens, qs):
        n = collections.Counter(); c = collections.Counter()
        for q in qs:
            for pat, e in groups[q].items():
                for m in ens:
                    if pat & bit[m]:
                        n[m] += e[0]; c[m] += e[1]
        return {m: (c[m] + 1) / (n[m] + 2) for m in ens}

    def fit_pattern(groups, qs, alpha=20.0):
        bypat = collections.defaultdict(lambda: [0, 0]); bycnt = collections.defaultdict(lambda: [0, 0])
        for q in qs:
            for pat, e in groups[q].items():
                bypat[pat][0] += e[1]; bypat[pat][1] += e[0]
                k = bin(pat).count("1")
                bycnt[k][0] += e[1]; bycnt[k][1] += e[0]
        cr = {k: (h + 1) / (t + 2) for k, (h, t) in bycnt.items()}
        return {p: (h + alpha * cr.get(bin(p).count("1"), .05)) / (t + alpha)
                for p, (h, t) in bypat.items()}, cr

    def best_threshold(groups, pats, fn, qs):
        best = None
        for t in sorted({fn(p) for p in pats}):
            keep = [p for p in pats if fn(p) >= t - 1e-12]
            v = statistics.mean(f1_from(groups[q], keep, ngold[q]) for q in qs)
            if best is None or v > best[0]:
                best = (v, t)
        return best[1]

    out = []
    for k in (3, 4, 5, 6):
        for ens in itertools.combinations(POOL, k):
            emask = 0
            for m in ens:
                emask |= bit[m]
            groups = {q: group(cand[q], emask) for q in qids}
            pats = [p for p in range(1, 1 << len(POOL)) if p & ~emask == 0]

            scored = collections.defaultdict(dict)
            # --- honest arms: fit on train, score on test
            for train, test in folds:
                w = fit_weights(groups, ens, train)
                pat, cr = fit_pattern(groups, train)
                rules = {
                    "count":    lambda p: bin(p).count("1"),
                    "uniform":  lambda p: 0.25 * bin(p).count("1"),
                    "weighted-cf": lambda p, _w=w: sum(_w[m] for m in ens if p & bit[m]),
                    "pattern-cf":  lambda p, _pt=pat, _cr=cr: _pt.get(p, _cr.get(bin(p).count("1"), .05)),
                }
                for arm, fn in rules.items():
                    th = best_threshold(groups, pats, fn, train)
                    keep = [p for p in pats if fn(p) >= th - 1e-12]
                    for q in test:
                        scored[arm][q] = f1_from(groups[q], keep, ngold[q])

            # --- oracle arms: fit on EVERYTHING (upper bound, no estimation error)
            w_all = fit_weights(groups, ens, qids)
            pat_all, cr_all = fit_pattern(groups, qids)
            for arm, fn in {
                "weighted-oracle": lambda p, _w=w_all: sum(_w[m] for m in ens if p & bit[m]),
                "pattern-oracle":  lambda p, _pt=pat_all, _cr=cr_all: _pt.get(p, _cr.get(bin(p).count("1"), .05)),
            }.items():
                th = best_threshold(groups, pats, fn, qids)
                keep = [p for p in pats if fn(p) >= th - 1e-12]
                for q in qids:
                    scored[arm][q] = f1_from(groups[q], keep, ngold[q])

            row = dict(k=k, ens=list(ens))
            for arm in scored:
                row[arm] = 100 * statistics.mean(scored[arm][q] for q in qids)
            bm = max(ens, key=lambda m: statistics.mean(singles[m][q] for q in qids))
            row["best"] = 100 * statistics.mean(singles[bm][q] for q in qids)
            pr = sorted((statistics.mean(singles[m][q] for q in qids) for m in ens), reverse=True)
            row["dominance"] = pr[0] / max(1e-9, statistics.mean(pr[1:]))
            out.append(row)
        print(f"    size {k} done", flush=True)

    print(f"\n  mean over ensembles, by size")
    print(f"  {'size':>5}{'n':>5}{'count':>9}{'uniform':>9}{'wgt-cf':>9}{'wgt-orc':>9}"
          f"{'pat-cf':>9}{'pat-orc':>9}")
    print("  " + "-" * 64)
    for k in (3, 4, 5, 6):
        R = [r for r in out if r["k"] == k]
        print(f"  {k:>5}{len(R):>5}" + "".join(
            f"{statistics.mean(r[a] for r in R):>9.2f}"
            for a in ["count", "uniform", "weighted-cf", "weighted-oracle",
                      "pattern-cf", "pattern-oracle"]))

    print(f"\n  GAIN OVER COUNTING  (the crossover under test)")
    print(f"  {'size':>5}{'wgt-cf':>10}{'wgt-ORACLE':>13}{'pat-cf':>10}{'pat-ORACLE':>13}")
    print("  " + "-" * 52)
    for k in (3, 4, 5, 6):
        R = [r for r in out if r["k"] == k]
        c = statistics.mean(r["count"] for r in R)
        print(f"  {k:>5}" + "".join(
            f"{statistics.mean(r[a] for r in R) - c:>+10.2f}" if a.endswith("cf")
            else f"{statistics.mean(r[a] for r in R) - c:>+13.2f}"
            for a in ["weighted-cf", "weighted-oracle", "pattern-cf", "pattern-oracle"]))

    bad = [r for r in out if abs(r["count"] - r["uniform"]) > 1e-9]
    print(f"\n  uniform == count sanity: "
          f"{'OK, identical everywhere' if not bad else f'!! DIFFERS on {len(bad)} ensembles — BUG'}")

    xs = [r["dominance"] for r in out]
    for a in ["weighted-cf", "weighted-oracle"]:
        ys = [r[a] - r["count"] for r in out]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
        print(f"  corr(dominance, {a} - count) = {num/den:+.3f}")

    json.dump(out, open(RESULTS / "crossover_controls.json", "w"), indent=1)
    print("\n  wrote results/crossover_controls.json")


if __name__ == "__main__":
    main()
