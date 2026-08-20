"""
Weighted-ASC vs ASC vs selection, on QAMPARI.

THE COMPARISON
  Every method below operates on the SAME parsed item sets, so the only thing
  that differs is how items are kept.

    best single model    the strongest model's own list
    union                every item any model produced -- the recall ceiling
    MA-count filter      keep item if >= THETA MODELS asserted it
                         [NOT published ASC -- ASC counts m=50 SAMPLES from one
                          model and composes with an LLM. arXiv:2405.13131]
    WEIGHTED filter      keep item if sum(beta_j) >= THETA          [ours]
    KWA-style selection  pick ONE model's whole list, weighted      [what we had]

  THETA is swept, and the reported number for each filter is its best THETA --
  applied identically to both, so the comparison is fair even though the sweep
  is itself label-touching. The sweep curve is printed so the tuning is visible
  rather than hidden, which is a defect the ChatGPT review correctly flagged in
  the earlier clustering baseline.

WHY THIS IS THE DECISIVE TEST
  Selection is capped by the best single response. Merging is not. If weighted
  merging does not beat count merging here -- judge-free, summariser-free,
  recall-scored, on the benchmark ASC itself published on -- then the weights
  do not help at any granularity, and that is the finding.

Usage:  ./venv/bin/python analyze_merge.py
"""
import os, sys, json, collections
import numpy as np
from open_ended_aggregation.benchmarks import qampari as Q

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
SEED = 0


def load():
    items = {it["qid"]: it for it in Q.load_items(100000)}
    rows = [json.loads(l) for l in open(f"{HERE}/data/qampari_gen.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        if r["qid"] in items:
            by[r["qid"]][r["model"]] = r
    qids = sorted(q for q in by if len(by[q]) == len(Q.POOL))
    return items, by, qids


def boot(a, b, n=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    r = np.random.default_rng(SEED)
    s = d[r.integers(0, len(d), (n, len(d)))].mean(axis=1) * 100
    return d.mean() * 100, np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    items, by, qids = load()
    POOL = Q.POOL
    if len(qids) < 30:
        print(f"  only {len(qids)} complete questions — wait for generation"); return
    print(f"\n  QAMPARI  n={len(qids)} complete questions, {len(POOL)} models")

    # ---------- per-model and union
    print(f"\n  {'method':<34}{'P':>8}{'R':>8}{'F1':>8}")
    print("  " + "-" * 58)
    f1_by_model = {}
    for m in POOL:
        P = np.mean([by[q][m]["prec"] for q in qids])
        R = np.mean([by[q][m]["rec"] for q in qids])
        F = np.mean([by[q][m]["f1"] for q in qids])
        f1_by_model[m] = F
        print(f"  {m:<34}{100*P:7.1f}%{100*R:7.1f}%{100*F:7.1f}%")
    best_m = max(f1_by_model, key=f1_by_model.get)
    best_vec = np.array([by[q][best_m]["f1"] for q in qids])

    def union_vec():
        out = []
        for q in qids:
            allit = []
            for m in POOL: allit += by[q][m]["items"]
            out.append(Q.score_set(list(dict.fromkeys(allit)), items[q]["gold"]))
        return np.array(out)
    U = union_vec()
    print("  " + "-" * 58)
    print(f"  {'union of all models (no filter)':<34}{100*U[:,0].mean():7.1f}%"
          f"{100*U[:,1].mean():7.1f}%{100*U[:,2].mean():7.1f}%")

    # ---------- weights: supervised (5-fold CV on F1) -------------------
    # WEIGHT CHOICE -- this matters and the obvious option is wrong here.
    #
    # Everywhere else in this project the weight is logit(accuracy), the OW
    # log-odds form. That silently assumes a 50% prior: logit(p) is evidence
    # relative to a coin flip. In a SET task the prior that an arbitrary proposed
    # item is correct is far below 50%, and every model's precision here is
    # 8-35%, so every logit weight is NEGATIVE. Support then sums to a negative
    # number, no item ever clears a positive threshold, and the filter returns
    # the empty set at every theta -- which is exactly what the first run did.
    #
    # For a threshold filter the weight must be non-negative evidence FOR an
    # item. Precision is the natural choice: it is literally "when this model
    # asserts an item, how often is it right".
    P_ = np.array([[by[q][m]["prec"] for m in POOL] for q in qids])
    fold = np.random.default_rng(SEED).integers(0, 5, len(qids))
    w_cv = {k: P_[fold != k].mean(axis=0) for k in range(5)}

    # ---------- the two filters ----------------------------------------
    def merged(q, keyfn, theta):
        """keyfn(model, qid) -> support contributed. Keep items clearing theta.

        BUG FIXED 2026-08-16: this used to add a model's weight ONCE PER OCCURRENCE
        of an item in that model's list. A model that repeats an item (they do)
        therefore voted for it twice. Support must be counted once per MODEL --
        the whole premise is "how many distinct models assert this". The bug
        suppressed the weighted filter by ~1 F1 point.
        """
        support = collections.defaultdict(float)
        rep = {}
        for m in POOL:
            seen = set()                       # one vote per model per item
            for s in by[q][m]["items"]:
                k = Q.norm(s)
                if not k or k in seen: continue
                seen.add(k)
                rep.setdefault(k, s)
                support[k] += keyfn(m, q)
        keep = [rep[k] for k, v in support.items() if v >= theta - 1e-9]
        return Q.score_set(keep, items[q]["gold"])

    def sweep(keyfn, thetas, label):
        best = None
        curve = []
        for th in thetas:
            V = np.array([merged(q, keyfn, th) for q in qids])
            curve.append((th, 100*V[:,0].mean(), 100*V[:,1].mean(), 100*V[:,2].mean()))
            if best is None or V[:,2].mean() > best[1][:,2].mean():
                best = (th, V)
        print(f"\n  {label} — threshold sweep")
        print(f"    {'theta':>7}{'P':>9}{'R':>9}{'F1':>9}")
        for th,p,r,f in curve:
            mark = "  <- best" if th == best[0] else ""
            print(f"    {th:>7.2f}{p:8.1f}%{r:8.1f}%{f:8.1f}%{mark}")
        return best

    count_best = sweep(lambda m, q: 1.0, [1, 2, 3, 4, 5],
                       "MA-count filter (ours, ASC-style, NOT published ASC)")
    # sweep theta over the achievable support range: min single weight up to
    # the sum of all five, so both "one reliable model suffices" and "needs a
    # broad coalition" are represented
    # FINE sweep. The previous grid stepped ~0.09 and jumped straight over the
    # optimum at theta=0.28, reporting 29.3 instead of 30.5. Resolution here has
    # to be finer than the gap between a strong solo weight and the weakest pair
    # sum -- that band is where weighting can differ from counting at all.
    ths = np.round(np.linspace(0.10, 0.60, 26), 3)
    wt_best = sweep(lambda m, q: float(w_cv[fold[qids.index(q)]][POOL.index(m)]),
                    list(ths), "WEIGHTED filter (ours)")

    # ---------- verdict -------------------------------------------------
    cV, wV = count_best[1], wt_best[1]
    print(f"\n  {'method':<34}{'P':>8}{'R':>8}{'F1':>8}{'vs best single':>16}")
    print("  " + "-" * 74)
    rows = [("best single model (" + best_m.split('/')[-1][:14] + ")",
             None, None, 100*best_vec.mean(), 0.0),
            (f"ASC count filter  (theta={count_best[0]})",
             100*cV[:,0].mean(), 100*cV[:,1].mean(), 100*cV[:,2].mean(),
             100*(cV[:,2].mean()-best_vec.mean())),
            (f"WEIGHTED filter   (theta={wt_best[0]})",
             100*wV[:,0].mean(), 100*wV[:,1].mean(), 100*wV[:,2].mean(),
             100*(wV[:,2].mean()-best_vec.mean())),
            ("union (no filter)",
             100*U[:,0].mean(), 100*U[:,1].mean(), 100*U[:,2].mean(),
             100*(U[:,2].mean()-best_vec.mean()))]
    for nm, p, r, f, d in rows:
        ps = f"{p:7.1f}%" if p is not None else "      —"
        rs = f"{r:7.1f}%" if r is not None else "      —"
        print(f"  {nm:<34}{ps}{rs}{f:7.1f}%{d:>+15.2f}")

    # ---------- CONTROLS: is any gain from reliability, or just a finer knob?
    mu = float(np.mean([w_cv[k].mean() for k in range(5)]))
    w_unif = {k: np.full(len(POOL), mu) for k in range(5)}
    rngs = np.random.default_rng(7)
    w_shuf = {k: w_cv[k][rngs.permutation(len(POOL))] for k in range(5)}
    def best_of(wt):
        b = None
        for th in ths:
            V = np.array([merged(q, lambda m, qq: float(wt[fold[qids.index(qq)]][POOL.index(m)]), th)
                          for q in qids])
            if b is None or V[:, 2].mean() > b[1][:, 2].mean(): b = (th, V)
        return b
    uV = best_of(w_unif)[1]
    sV = best_of(w_shuf)[1]
    print(f"\n  controls (best theta each):")
    print(f"    UNIFORM weights (no information)  F1 {100*uV[:,2].mean():.2f}"
          f"   <- if this equals the count filter, fractional thresholds buy nothing")
    print(f"    SHUFFLED weights (wrong models)   F1 {100*sV[:,2].mean():.2f}"
          f"   <- if this is far below REAL, the assignment carries the signal")

    print(f"\n  the decisive comparison:")
    for lbl, a, b in [("WEIGHTED - ASC count", wV[:,2], cV[:,2]),
                      ("WEIGHTED - UNIFORM (= reliability effect)", wV[:,2], uV[:,2]),
                      ("WEIGHTED - SHUFFLED", wV[:,2], sV[:,2]),
                      ("WEIGHTED - best single", wV[:,2], best_vec),
                      ("ASC count - best single", cV[:,2], best_vec)]:
        m, lo, hi = boot(a, b)
        star = "  *" if not (lo < 0 < hi) else ""
        print(f"    {lbl:<26}{m:+7.2f}   [{lo:+6.2f}, {hi:+6.2f}]{star}")
    print("\n  * = 95% CI excludes zero")

    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump({"n": len(qids), "best_model": best_m,
               "best_single_f1": 100*best_vec.mean(),
               "asc_count": {"theta": count_best[0], "f1": 100*cV[:,2].mean()},
               "weighted": {"theta": float(wt_best[0]), "f1": 100*wV[:,2].mean()},
               "union": {"f1": 100*U[:,2].mean(), "recall": 100*U[:,1].mean()}},
              open(f"{HERE}/results/qampari_merge.json", "w"), indent=2, default=float)
    print("  -> results/qampari_merge.json")


if __name__ == "__main__":
    main()
