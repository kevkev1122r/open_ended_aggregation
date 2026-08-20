"""
QAMPARI: weighted multi-agent merging against the standard aggregation baselines.

  ./venv/bin/python compare_methods.py [path_to_gen_jsonl]

Defaults to data/qampari_gen.jsonl. Pass data/qampari_gen.jsonl.n198.bak to get
the frozen n=198 state. Re-run as generation grows -- n is printed, and the
comparison is only meaningful against a stated n.

Method definitions, all on the SAME parsed item sets so only the rule differs:
  mean / best single    no aggregation
  OW selection          reliability-weighted vote over whole RESPONSES, the
                        published Optimal-Weights rule. With global weights and
                        open-ended answers no two responses coincide, so the
                        argmax is always the highest-weighted model -- it cannot
                        do anything but pick one model.
  OW log-odds items     OW's weight form log(x/(1-x)) applied per item. Every
                        QAMPARI precision is < 0.5 so every weight is negative
                        and more agreement means LESS support. Reported to show
                        the form is unusable here, not as a serious arm.
  union / MV / MA-count count filter at theta = 1 / 3 / 2 over MODELS.

                        NOT ASC. Published ASC (arXiv:2405.13131) draws m=50
                        stochastic samples from ONE model, tunes theta on a
                        validation set, and composes the survivors with an LLM.
                        Counting over models instead of samples measures
                        inter-model agreement, not self-consistency -- a
                        different estimand. This arm is an unpublished
                        multi-agent adaptation and must never be reported as ASC.

  MA-count + OW         the same filter with weights = cross-fitted per-model
                        precision (non-negative). Ours.
  oracle selection      best single response per question; ceiling on anything
                        that picks rather than merges
"""
import sys, os, json, math, collections, statistics, zipfile, random

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
from open_ended_aggregation.analysis.verify_qampari import norm, score_set, bootstrap


def load(path):
    recs = [json.loads(l) for l in open(path)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]][r["model"]] = r
    models = sorted({r["model"] for r in recs})
    qids = sorted(q for q, d in byq.items() if len(d) == len(models))
    z = zipfile.ZipFile(f"{HERE}/data/qampari.zip")
    gold, want = {}, set(qids)
    with z.open("qampari_data/dev_data.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r["qid"] in want:
                gold[r["qid"]] = [
                    {norm(a) for a in ([g["answer_text"]] + list(g.get("aliases") or []))
                     if str(a).strip()} for g in r["answer_list"]]
    return byq, models, qids, gold


def main(path):
    byq, models, qids, gold = load(path)
    n = len(qids)
    glob = {m: statistics.mean(byq[q][m]["prec"] for q in qids) for m in models}

    idx = list(range(n)); random.Random(0).shuffle(idx)
    fold_of = {qids[i]: j % 5 for j, i in enumerate(idx)}
    wf = {f: {m: statistics.mean(byq[q][m]["prec"] for q in qids if fold_of[q] != f)
              for m in models} for f in range(5)}

    def keys(q, m):
        out, seen = [], set()
        for it in byq[q][m]["items"]:
            k = norm(it)
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
            out.append(score_set([rep[k] for k, v in acc.items()
                                  if v >= theta - 1e-12], gold[q]))
        return out

    ONE = lambda q: {m: 1.0 for m in models}
    CF = lambda q: wf[fold_of[q]]

    singles = {m: [score_set(byq[q][m]["items"], gold[q]) for q in qids] for m in models}
    best_m = max(models, key=lambda m: statistics.mean(singles[m]))
    best = singles[best_m]
    mean_single = [statistics.mean(singles[m][i] for m in models) for i in range(n)]

    ow_pick = max(models, key=lambda m: glob[m])
    lo = {m: math.log(max(glob[m], 1e-9) / (1 - min(glob[m], 1 - 1e-9))) for m in models}
    ow_item = max(((statistics.mean(filt(lambda q: lo, t)), t)
                   for t in [-40 + 0.25 * i for i in range(400)]), key=lambda r: r[0])
    ours = max(((statistics.mean(filt(CF, t)), t)
                for t in [round(0.05 + 0.005 * i, 4) for i in range(231)]), key=lambda r: r[0])

    rows = [
        ("mean single model", mean_single, ""),
        (f"best single model ({best_m})", best, ""),
        ("OW - response selection", singles[ow_pick], f"argmax weight = {ow_pick}"),
        ("OW - log-odds item filter", filt(lambda q: lo, ow_item[1]),
         f"best theta {ow_item[1]:.1f} = filter off"),
        ("union / no filter", filt(ONE, 1), "theta=1"),
        ("MV - strict majority", filt(ONE, 3), "theta=3"),
        ("MA-count filter (ours, ASC-style)", filt(ONE, 2), "theta=2, tuned on eval"),
        ("MA-count + OW (ours)", filt(CF, ours[1]), f"theta={ours[1]:.3f}"),
        ("oracle selection (needs labels)",
         [max(singles[m][i] for m in models) for i in range(n)], "selection ceiling"),
    ]

    print(f"\n  {os.path.basename(path)}   n={n} complete questions, {len(models)} models")
    print("  NOTE: no arm here is published ASC. See module docstring.")
    print(f"  {'method':<34}{'F1':>7}{'vs best single':>27}   note")
    print("  " + "-" * 100)
    for name, sc, note in rows:
        f1 = statistics.mean(sc) * 100
        if name.startswith("best single"):
            cell = "(reference)"
        else:
            d, l, h = bootstrap(sc, best)
            cell = f"{d:+7.2f}  [{l:+6.2f},{h:+6.2f}]" + (" *" if (l > 0 or h < 0) else "")
        print(f"  {name:<34}{f1:7.2f}{cell:>27}   {note}")
    print("\n  head-to-head")
    d = dict((r[0], r[1]) for r in rows)
    for lbl, a, b in [("MA-count+OW - MA-count", "MA-count + OW (ours)", "MA-count filter (ours, ASC-style)"),
                      ("MA-count+OW - MV", "MA-count + OW (ours)", "MV - strict majority"),
                      ("MA-count    - MV", "MA-count filter (ours, ASC-style)", "MV - strict majority")]:
        m_, l, h = bootstrap(d[a], d[b])
        print(f"    {lbl:<26}{m_:+7.2f}  [{l:+6.2f},{h:+6.2f}]" + ("  *" if (l > 0 or h < 0) else ""))
    print("\n  * = 95% CI excludes 0 (paired bootstrap, 10k, over questions)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/data/qampari_gen.jsonl")
