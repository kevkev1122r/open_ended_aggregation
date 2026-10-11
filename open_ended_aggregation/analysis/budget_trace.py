"""
What the budget actually does, traced candidate by candidate on one question.

WHY THIS EXISTS
  "How is k derived?" was the question the 22 Aug meeting could not answer from
  the code, and it is the piece of the method most often described wrongly. The
  short version is that k is not tuned, not a hyperparameter, and not a constant.
  It is recomputed per question as the argmax of an expected-F1 curve.

  This prints that curve for a real question so the argmax is visible rather than
  asserted.

THE RULE, DERIVED
  Per-question F1 with k kept, H of them correct, G gold answers:

      P = H/k,  R = H/G,  F1 = 2PR/(P+R) = 2H/(k+G)

  H is unknown at test time, but the scorer emits p_i = P(claim i correct), so
  the expected hits in the top k is the cumulative sum of the top k probabilities.
  Substituting that and the predicted gold count Ghat:

      k*(q) = argmax_k  2 * sum_{i<=k} p_i / (k + Ghat_q)

  which is analysis/weighting_v2.py:116 exactly.

PROTOCOL
  Fits on 4 folds, traces a question from the held-out fifth, so nothing shown
  here was trained on. Logistic only -- no GBM, since boosting is out of the
  method as of the 23 Aug handoff.

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.budget_trace
  ./venv/bin/python -u -m open_ended_aggregation.analysis.budget_trace --qid 53__wikitables_composition__dev
"""
import math, json, random, argparse, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    fit_logistic, fit_pattern, SEED, FOLDS)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--rows", type=int, default=16, help="candidates to print")
    ap.add_argument("--json", default=None,
                    help="also dump the full trace to this path, for figures")
    a = ap.parse_args()

    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    # pick a question with enough candidates to make the curve interesting
    if a.qid and a.qid in qids:
        qi = qids.index(a.qid)
    else:
        best = None
        for j, q in enumerate(qids):
            m = int((rowq == j).sum())
            if 20 <= m <= 60 and 3 <= ngold[q] <= 12:
                if best is None or abs(m - 35) < abs(best[1] - 35):
                    best = (j, m)
        qi = best[0]
    q = qids[qi]
    f = qfold[qi]

    tr_q = np.nonzero(qfold != f)[0]
    tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
    trm = tm[rowq]

    plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
    pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
    X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0
    wl = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
    p = 1.0 / (1.0 + np.exp(-np.clip(X[:, LEAN] @ wl, -30, 30)))

    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for j, qq in enumerate(qids):
        w = np.nonzero(rowq == j)[0]
        QF[j, :4] = [1.0, math.log1p(len(w)) / 5.0,
                     (cnt[w].max() if len(w) else 0) / n, (cnt[w] >= 2).sum() / 20.0]
        QF[j, 4 + tix.index(qtype[qq])] = 1.0
    QG = np.zeros((nq, QF.shape[1] + 3))
    QG[:, :QF.shape[1]] = QF
    for j in range(nq):
        w = np.nonzero(rowq == j)[0]
        QG[j, QF.shape[1]:] = [p[w].sum() / 10.0, p[w][cnt[w] >= 2].sum() / 10.0,
                               float(p[w].max()) if len(w) else 0.0]
    beta = fit_gold_ridge(QG[tr_q], np.log(np.maximum(ng_arr[tr_q], 1.0)))   # log-scale answer count
    Ghat = max(float(np.exp(QG[qi] @ beta)), 1.0)

    w = np.nonzero(rowq == qi)[0]
    o = w[np.argsort(-p[w])]
    G = ngold[q]

    print("=" * 96)
    print("  BUDGET TRACE -- how k is chosen for one question")
    print("=" * 96)
    print(f"\n  question   {qtext[q][:74]}")
    print(f"  qid        {q}")
    print(f"  fold       {f} (held out -- nothing below was trained on)")
    print(f"  candidates {len(w)}     true gold count G = {G}     "
          f"predicted Ghat = {Ghat:.2f}")

    print(f"\n  {'k':>4}{'candidate':<44}{'p':>8}{'sum p':>9}"
          f"{'2*sum/(k+Ghat)':>17}{'correct?':>10}{'true F1':>10}")
    print("  " + "-" * 94)
    cs = 0.0; hits = 0; seen = set()
    vals, tf = [], []
    for kk, i in enumerate(o, 1):
        cs += p[i]
        ok = gis[i] is not None
        if ok:
            hits += 1; seen.add(gis[i])
        val = 2 * cs / (kk + Ghat)
        prec = hits / kk; rec = len(seen) / max(1, G)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        vals.append(val); tf.append(f1)
    kstar = int(np.argmax(vals)) + 1
    kbest = int(np.argmax(tf)) + 1

    cs = 0.0; hits = 0; seen = set()
    for kk, i in enumerate(o, 1):
        cs += p[i]
        ok = gis[i] is not None
        if ok:
            hits += 1; seen.add(gis[i])
        if kk > a.rows and kk not in (kstar, kbest):
            continue
        key = cand[q][0]
        name = [c[3] for c in cand[q] if True][0]
        lbl = ""
        if kk == kstar:
            lbl += " <= CHOSEN"
        if kk == kbest:
            lbl += " <= true optimum"
        txt = [c[3] for c in cand[q]][list(w).index(i)]
        print(f"  {kk:>4}{txt[:42]:<44}{p[i]:>8.3f}{cs:>9.2f}"
              f"{vals[kk-1]:>17.4f}{('yes' if ok else '-'):>10}"
              f"{tf[kk-1]:>10.3f}{lbl}")

    print(f"\n  chosen k = {kstar}   -> actual F1 {100*tf[kstar-1]:.2f}")
    print(f"  best possible k = {kbest} -> actual F1 {100*tf[kbest-1]:.2f}   "
          f"(oracle, needs the labels)")
    print(f"  keep-everything k = {len(w)} -> actual F1 {100*tf[-1]:.2f}")
    print(f"\n  the curve being maximised uses only p and Ghat. the 'true F1' "
          f"column is\n  never seen by the rule -- it is printed to show how "
          f"close the argmax lands.")

    if a.json:
        names = [c[3] for c in cand[q]]
        ranks = [c[2] for c in cand[q]]
        pats_q = [c[0] for c in cand[q]]
        wl_ = list(w)
        rows = []
        for kk, i in enumerate(o, 1):
            j = wl_.index(i)
            rows.append({
                "k": kk,
                "candidate": names[j],
                "asserted": {m: int(ranks[j][m]) + 1 for m in pats_q[j]},
                "support": len(pats_q[j]),
                "p": float(p[i]),
                "correct": gis[i] is not None,
                "plugin": float(vals[kk - 1]),
                "true_f1": float(tf[kk - 1]),
            })
        payload = {
            "qid": q, "question": qtext[q], "fold": int(f),
            "models": list(models),
            "list_lengths": {m: int(nlist[q][m]) for m in models},
            "gold_count": int(G), "ghat": float(Ghat),
            "n_candidates": len(w),
            "k_chosen": kstar, "k_oracle": kbest,
            "f1_chosen": float(100 * tf[kstar - 1]),
            "f1_oracle": float(100 * tf[kbest - 1]),
            "f1_keep_all": float(100 * tf[-1]),
            "rows": rows,
        }
        with open(a.json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
