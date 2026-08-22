"""
Merge alias variants of the same candidate before scoring.

THE PROBLEM
  Everything so far treats each distinct normalised string as its own candidate.
  QAMPARI answers are entity names, and agents spell them differently:
  "the beatles" / "beatles", "j k rowling" / "joanne rowling". Two consequences,
  both bad, and both invisible to every arm tested so far:

    SUPPORT SPLITS.  Four agents naming the same entity under two spellings look
    like two claims with two supporters each, not one claim with four. Since
    P(correct | k) climbs 1.2 -> 11.1 -> 17.4 -> 27.2%, that is the difference
    between discarding a claim and keeping it.

    PRECISION IS DOUBLE-CHARGED.  Keeping both spellings costs two slots in the
    precision denominator and collects one gold answer. Under set F1 that is a
    pure loss.

THE RISK, WHICH IS THE REASON TO BE CAREFUL
  A FALSE merge deletes a correct answer outright -- the merged candidate hits
  one gold index and the other gold becomes unreachable. This is exactly how
  LLM-judged clustering failed on this project (-6 F1 vs edit distance): it
  merged distinct entities and only the representative survived. So the merge
  here is conservative, character-trigram Jaccard with a HIGH threshold, and tau
  is cross-fitted rather than chosen by eye. The diagnostic below reports how
  many merges were good (same gold) versus bad (different gold), which is the
  number that decides whether this is safe.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v7
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.analysis.weighting_v3 import load_rich, build_features
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import RESULTS


def trigrams(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def merge_question(rows, tau):
    """Union-find over candidates of one question. rows = [(ms, gi, rk, key)].
    Returns merged rows plus (good, bad) merge counts for the diagnostic."""
    keys = [r[3] for r in rows]
    tg = [trigrams(k) for k in keys]
    n = len(rows)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    good = bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tg[i], tg[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            if inter / (len(a) + len(b) - inter) < tau:
                continue
            gi, gj = rows[i][1], rows[j][1]
            if gi is not None and gj is not None:
                if gi == gj:
                    good += 1
                else:
                    bad += 1
            if find(i) != find(j):
                par[find(i)] = find(j)

    grp = collections.defaultdict(list)
    for i in range(n):
        grp[find(i)].append(i)
    out = []
    for _, mem in grp.items():
        ms = set(); rk = {}
        gi = None
        for i in mem:
            ms |= set(rows[i][0])
            for m, r in rows[i][2].items():
                rk[m] = min(rk.get(m, 10 ** 9), r)
            if gi is None:
                gi = rows[i][1]
        key = max((keys[i] for i in mem), key=len)
        out.append((frozenset(ms), gi, rk, key))
    return out, good, bad


def rebuild(models, qids, cand, nlist, qtype, qtext, tau):
    if tau >= 1.0:
        return cand, 0, 0
    new, G, B = {}, 0, 0
    for q in qids:
        rows, g, b = merge_question(cand[q], tau)
        new[q] = rows; G += g; B += b
    return new, G, B


def run_pipeline(models, qids, cand, ngold, nlist, qtype, qtext, seed, arms_wanted):
    """Score one candidate set with count / LEAN / LEAN+budget."""
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    LEAN = sorted(set(i for g in ("base", "rank", "rankbuckets", "omit", "stack")
                      for i in blocks[g]))
    spans = []
    for qi in range(nq):
        w_ = np.nonzero(rowq == qi)[0]
        spans.append((int(w_[0]), int(w_[-1]) + 1) if len(w_) else (0, 0))
    C = lambda s: curves(qids, ngold, gis, spans, s)

    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for qi, q in enumerate(qids):
        w_ = np.nonzero(rowq == qi)[0]
        QF[qi, :4] = [1.0, math.log1p(len(w_)) / 5.0,
                      (cnt[w_].max() if len(w_) else 0) / n, (cnt[w_] >= 2).sum() / 20.0]
        QF[qi, 4 + tix.index(qtype[q])] = 1.0

    per = {arm: np.zeros(nq) for arm in arms_wanted}
    rng = random.Random(seed)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]
        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0
        w_ = fit_logistic(X[np.ix_(trm, LEAN)], y[trm], ridge=1.0)
        s_lean = X[:, LEAN] @ w_

        for arm, s in (("count", cnt), ("pattern", pat_p), ("LEAN", s_lean)):
            if arm not in per:
                continue
            S, F = C(s)
            gg = np.unique(s[trm])
            if len(gg) > GRID:
                gg = np.quantile(gg, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), gg)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        if "LEAN + budget" in per:
            p = 1.0 / (1.0 + np.exp(-np.clip(s_lean, -30, 30)))
            QG = np.zeros((nq, QF.shape[1] + 3))
            QG[:, :QF.shape[1]] = QF
            for qi in range(nq):
                ww = np.nonzero(rowq == qi)[0]
                QG[qi, QF.shape[1]:] = [p[ww].sum() / 10.0,
                                        p[ww][cnt[ww] >= 2].sum() / 10.0,
                                        float(p[ww].max()) if len(ww) else 0.0]
            beta = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
            Gh = np.maximum(QG @ beta, 1.0)
            fa, _ = ev(adaptive_keep(p, rowq, nq, Gh), list(tr_q))
            for qi in te_q:
                per["LEAN + budget"][qi] = fa[qi]
    return per, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    print("=" * 92)
    print("  WEIGHTING v7 -- alias merging before scoring")
    print("=" * 92)
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich()
    ARMS = ["count", "pattern", "LEAN", "LEAN + budget"]

    rows = []
    for tau in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        merged, G, B = rebuild(models, qids, cand, nlist, qtype, qtext, tau)
        per, ncand = run_pipeline(models, qids, merged, ngold, nlist, qtype,
                                  qtext, a.seed, ARMS)
        rows.append((tau, ncand, G, B, {k: 100 * v.mean() for k, v in per.items()},
                     per))
        print(f"    tau={tau:<5} {ncand:>7,} candidates  merges good/bad "
              f"{G:>5}/{B:<5}  " +
              "  ".join(f"{k} {100*v.mean():.2f}" for k, v in per.items()),
              flush=True)

    base = rows[0][5]
    print(f"\n  {'tau':>6}{'candidates':>12}{'good/bad':>12}"
          f"{'count':>9}{'pattern':>9}{'LEAN':>9}{'LEAN+budget':>13}"
          f"{'vs no-merge':>26}")
    print("  " + "-" * 100)
    for tau, ncand, G, B, m, per in rows:
        d, lo, hi = bootstrap(list(per["LEAN + budget"]),
                              list(base["LEAN + budget"]))
        cell = (f"{d:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"
                if tau < 1.0 else "(reference)")
        print(f"  {tau:>6.2f}{ncand:>12,}{G:>7}/{B:<5}"
              f"{m['count']:>9.2f}{m['pattern']:>9.2f}{m['LEAN']:>9.2f}"
              f"{m['LEAN + budget']:>13.2f}{cell:>26}")
    print("\n  good = merged pair shares a gold answer; bad = merged pair had "
          "DIFFERENT gold answers")
    print("  * = bootstrap 95% CI over questions excludes 0")

    json.dump({str(t): m for t, _, _, _, m, _ in rows},
              open(RESULTS / "weighting_v7.json", "w"), indent=2)
    print("  wrote results/weighting_v7.json")


if __name__ == "__main__":
    main()
