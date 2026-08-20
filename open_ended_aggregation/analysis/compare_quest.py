"""
QUEST: the same aggregation arms as compare_methods.py, on the second
set-valued benchmark.

WHY THIS FILE EXISTS SEPARATELY
  QUEST has its own loader, gold format (a bare list of Wikipedia titles, no
  aliases) and its own metric set (P/R/F1/Recall-5). Coupling it to the QAMPARI
  script is what produced the earlier "ASQA cannot replicate" confusion, so the
  datasets stay separate by design.

ARMS (identical rules, only the keep-decision differs)
  mean / best single      no aggregation
  OW - response selection the published Optimal-Weights rule; with global weights
                          and open-ended answers it always returns the
                          highest-weighted model
  union / MV / MA-count   count filter over MODELS at theta = 1 / 3 / 2
                          NOT ASC -- published ASC counts m=50 samples from ONE
                          model and composes with an LLM (arXiv:2405.13131)
  MA-count + OW           the same filter, weights = cross-fitted per-model
                          precision (non-negative). Ours.
  oracle selection        best single response per query; selection ceiling

Usage:  ./venv/bin/python compare_methods_quest.py [n]
"""
import sys, os, json, math, collections, statistics, random

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
from open_ended_aggregation.benchmarks import quest as R
from open_ended_aggregation.analysis.verify_qampari import bootstrap


def load(n=400):
    recs = [json.loads(l) for l in open(R.GEN_PATH)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({r["model"] for r in recs})
    items = {it["qid"]: it for it in R.load_items(n)}
    qids = sorted(q for q, d in byq.items() if len(d) == len(models) and q in items)
    return byq, models, qids, items


def main(n=400):
    byq, models, qids, items = load(n)
    N = len(qids)
    if N < 20:
        print(f"  only {N} complete queries -- too few, wait for generation"); return
    glob = {m: statistics.mean(byq[q][m]["prec"] for q in qids) for m in models}

    idx = list(range(N)); random.Random(0).shuffle(idx)
    fold = {qids[i]: j % 5 for j, i in enumerate(idx)}
    wf = {f: {m: statistics.mean(byq[q][m]["prec"] for q in qids if fold[q] != f)
              for m in models} for f in range(5)}

    def keys(q, m):
        out, seen = [], set()
        for it in byq[q][m]["items"]:
            k = R.norm(it)
            if k and k not in seen:
                seen.add(k); out.append((k, it))
        return out

    def filt(wfn, theta):
        out = []
        for q in qids:
            w = wfn(q); acc = collections.defaultdict(float); rep = {}
            for m in models:
                for k, it in keys(q, m):
                    acc[k] += w[m]; rep.setdefault(k, it)
            keep = [rep[k] for k, v in acc.items() if v >= theta - 1e-12]
            out.append(R.score_set(keep, items[q]["gold"]))
        return out

    ONE = lambda q: {m: 1.0 for m in models}
    CF = lambda q: wf[fold[q]]
    F1 = lambda rows: [r[2] for r in rows]          # score_set -> (p,r,f1,r5,f15)

    singles = {m: [R.score_set(byq[q][m]["items"], items[q]["gold"]) for q in qids]
               for m in models}
    best_m = max(models, key=lambda m: statistics.mean(F1(singles[m])))
    best = F1(singles[best_m])
    ow_pick = max(models, key=lambda m: glob[m])
    lo = {m: math.log(max(glob[m], 1e-9) / (1 - min(glob[m], 1 - 1e-9))) for m in models}
    ow_item = max(((statistics.mean(F1(filt(lambda q: lo, t))), t)
                   for t in [-40 + 0.5 * i for i in range(200)]), key=lambda r: r[0])
    ours = max(((statistics.mean(F1(filt(CF, t))), t)
                for t in [round(0.02 + 0.005 * i, 4) for i in range(240)]), key=lambda r: r[0])

    rows = [
        ("mean single model",
         [statistics.mean(F1(singles[m])[i] for m in models) for i in range(N)], ""),
        (f"best single ({best_m})", best, ""),
        ("OW - response selection", F1(singles[ow_pick]), f"argmax weight = {ow_pick}"),
        ("OW - log-odds item filter", F1(filt(lambda q: lo, ow_item[1])),
         f"theta {ow_item[1]:.1f}"),
        ("union / no filter", F1(filt(ONE, 1)), "theta=1"),
        ("MV - strict majority", F1(filt(ONE, len(models) // 2 + 1)),
         f"theta={len(models)//2+1}"),
        ("MA-count filter (ours, ASC-style)", F1(filt(ONE, 2)), "theta=2"),
        ("MA-count + OW (ours)", F1(filt(CF, ours[1])), f"theta={ours[1]:.3f}"),
        ("oracle selection (needs labels)",
         [max(F1(singles[m])[i] for m in models) for i in range(N)], "selection ceiling"),
    ]

    print(f"\n  QUEST  n={N} complete queries, {len(models)} agents")
    print("  NOTE: no arm here is published ASC. See module docstring.")
    print(f"  {'method':<36}{'F1':>7}{'vs best single':>27}   note")
    print("  " + "-" * 100)
    for name, sc, note in rows:
        f1 = statistics.mean(sc) * 100
        if name.startswith("best single"):
            cell = "(reference)"
        else:
            d, l, h = bootstrap(sc, best)
            cell = f"{d:+7.2f}  [{l:+6.2f},{h:+6.2f}]" + (" *" if (l > 0 or h < 0) else "")
        print(f"  {name:<36}{f1:7.2f}{cell:>27}   {note}")

    d = dict((r[0], r[1]) for r in rows)
    print("\n  head-to-head")
    for lbl, a, b in [("MA-count+OW - MA-count", "MA-count + OW (ours)", "MA-count filter (ours, ASC-style)"),
                      ("MA-count+OW - MV", "MA-count + OW (ours)", "MV - strict majority"),
                      ("MA-count    - MV", "MA-count filter (ours, ASC-style)", "MV - strict majority")]:
        m_, l, h = bootstrap(d[a], d[b])
        print(f"    {lbl:<26}{m_:+7.2f}  [{l:+6.2f},{h:+6.2f}]" + ("  *" if (l > 0 or h < 0) else ""))
    print("\n  * = 95% CI excludes 0 (paired bootstrap, 10k, over queries)")

    print(f"\n  per-agent (F1 / prec / rec / items)")
    for m in sorted(models, key=lambda m: -statistics.mean(F1(singles[m]))):
        print(f"    {m:<34}{100*statistics.mean(F1(singles[m])):6.2f}"
              f"{100*statistics.mean(byq[q][m]['prec'] for q in qids):7.2f}"
              f"{100*statistics.mean(byq[q][m]['rec'] for q in qids):7.2f}"
              f"{statistics.mean(len(byq[q][m]['items']) for q in qids):7.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
