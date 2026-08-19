"""
The aggregation comparison that analyze_domains.py was supposed to run.

WHY THIS FILE EXISTS
  analyze_domains.aggregation() compares answers by exact string identity of the
  last 120 characters. On this data 94.8% of questions have all seven answers as
  distinct strings, so there is no vote to take: every "majority vote" is a
  7-way tie broken at random, and every "weighted vote" degenerates to picking
  whichever single model carries the largest weight. Measured consequences:

      majority vote      81.13  ==  mean model accuracy (i.e. random choice)
      global weights     83.61  ==  always pick the globally best model
      per-domain weights 84.33  ==  pick the best model per domain

  None of that uses consensus, and none of it involves the similarity kernel --
  which is the actual contribution. analyze_domains never imports kernel_agg.

  This file supplies the missing piece: embed the answers, build a real
  similarity matrix, and run the aggregators from kernel_agg on it.

WHAT IS COMPARED
  majority (exact)       the degenerate baseline above, for reference
  medoid / cluster       similarity clustering, UNWEIGHTED  -- the kernel alone
  OW (exact)             the paper's Algorithm 1            -- the weights alone
  KWA (global beta)      kernel + one weight per model
  KWA (per-domain beta)  kernel + per-domain weights        -- the live hypothesis
  best single model      the bar that matters for deployment
  ceiling                any model correct

  The handoff's §2 refutation was that KWA never beats the better of its two
  ingredients. Rows 2 and 3 are those ingredients, so that claim is testable here.

All weights are 5-fold cross-validated: weights for a question come only from
folds that do not contain it. In-sample per-domain weighting was worth +1.86
points of pure optimism on this data.
"""
import os, sys, json
import numpy as np, pandas as pd
import kernel_agg as K
import analyze_domains as A
import benchmarks as B

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.90          # cluster threshold, as in kernel_agg.agg_majority_cluster
ENCODER = "sentence-transformers/all-mpnet-base-v2"


def final_answers(d, piv, models):
    """Embed the FINAL ANSWER, not the whole chain -- otherwise similarity is
    dominated by reasoning style and every long answer looks like every other."""
    resp = d.pivot_table(index="qid", columns="model", values="resp",
                         aggfunc="first").reindex(piv.index)[models]
    out = {}
    for qid in piv.index:
        out[qid] = [B.final_line(str(resp.loc[qid, m])) for m in models]
    return out


def embed_all(ans_by_q, qids, models):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    flat, index = [], {}
    for qid in qids:
        for j, a in enumerate(ans_by_q[qid]):
            index[(qid, j)] = len(flat)
            flat.append(a if a.strip() else "(no answer)")
    print(f"  embedding {len(flat):,} answers with {ENCODER}", flush=True)
    V = enc.encode(flat, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    return V, index


def run():
    d = A.load(gen="data/v2.jsonl", judged="data/v2_judged.jsonl")
    piv, dom = A.complete_matrix(d)
    models = list(piv.columns)
    qids = list(piv.index)
    doms = sorted(dom.unique())
    ok = piv[models]

    ans_by_q = final_answers(d, piv, models)
    V, index = embed_all(ans_by_q, qids, models)

    # ---- cross-validated weights, same folds as analyze_domains
    def logit_w(a):
        a = np.clip(a, 1e-3, 1 - 1e-3)
        return np.log(a / (1 - a))
    Kf = 5
    fold = pd.Series(np.random.default_rng(0).integers(0, Kf, len(piv)), index=piv.index)
    wg, wd = {}, {}
    for k in range(Kf):
        tr = piv[fold != k]
        wg[k] = logit_w(tr.mean().values)
        td = dom.loc[tr.index]
        wd[k] = {}
        for dd in doms:
            b = tr[td == dd]
            wd[k][dd] = logit_w(b.mean().values) if len(b) >= 20 else wg[k]
    foldof, domof = fold.to_dict(), dom.to_dict()

    methods = ["majority (exact)", "medoid / cluster (kernel only)",
               "OW exact (weights only)", "KWA (global beta)",
               "KWA (per-domain beta)"]
    hits = {m: [] for m in methods}
    rng = np.random.default_rng(0)

    for qid in qids:
        idx = [index[(qid, j)] for j in range(len(models))]
        E = V[idx]                       # 7 x dim, unit-normalised
        S = np.clip(E @ E.T, -1.0, 1.0)  # cosine similarity, 7 x 7
        # candidates ARE the observed answers, so ans is just 0..6 and S is the
        # candidate-by-candidate matrix agg_* expects
        ans = np.arange(len(models))
        k, dd = foldof[qid], domof[qid]

        picks = {
            "majority (exact)":               K.agg_majority_exact(ans, S, rng),
            "medoid / cluster (kernel only)": K.agg_majority_cluster(ans, S, rng, tau=TAU),
            "OW exact (weights only)":        K.agg_ow_exact(ans, S, rng, weights=wg[k]),
            "KWA (global beta)":              K.agg_kernel(ans, S, rng, betas=wg[k],
                                                           support="observed"),
            "KWA (per-domain beta)":          K.agg_kernel(ans, S, rng, betas=wd[k][dd],
                                                           support="observed"),
        }
        for m, p in picks.items():
            hits[m].append(bool(ok.loc[qid, models[p]]))

    res = {m: 100 * np.mean(v) for m, v in hits.items()}
    best_m = piv.mean().idxmax()
    res["best single model"] = 100 * piv.mean().max()
    res["ceiling (any correct)"] = 100 * piv.any(axis=1).mean()

    bs_vec = ok[best_m].values.astype(bool)

    def boot(a, b, n=10000):
        diff = np.asarray(a, float) - np.asarray(b, float)
        r = np.random.default_rng(0)
        s = diff[r.integers(0, len(diff), (n, len(diff)))].mean(axis=1) * 100
        return diff.mean() * 100, np.percentile(s, 2.5), np.percentile(s, 97.5)

    print(f"\n  n = {len(qids)} questions, {len(models)} models, best single = {best_m}")
    print(f"  {'method':<34}{'accuracy':>10}{'vs best single':>16}")
    print("  " + "-" * 62)
    for m in methods:
        print(f"  {m:<34}{res[m]:>10.2f}{res[m]-res['best single model']:>+16.2f}")
    print("  " + "-" * 62)
    for m in ["best single model", "ceiling (any correct)"]:
        print(f"  {m:<34}{res[m]:>10.2f}{res[m]-res['best single model']:>+16.2f}")

    print(f"\n  paired bootstrap vs best single model (95% CI):")
    for m in methods:
        mu, lo, hi = boot(hits[m], bs_vec)
        print(f"    {m:<34}{mu:>+8.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]"
              + ("  *" if not (lo < 0 < hi) else ""))

    print(f"\n  KWA vs its own two ingredients (the §2 refutation, retested):")
    for lbl, base in [("vs medoid/cluster (kernel only)", "medoid / cluster (kernel only)"),
                      ("vs OW exact (weights only)",      "OW exact (weights only)")]:
        mu, lo, hi = boot(hits["KWA (per-domain beta)"], hits[base])
        print(f"    KWA(per-domain) {lbl:<34}{mu:>+8.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]"
              + ("  *" if not (lo < 0 < hi) else ""))
    print("\n  * = 95% CI excludes zero")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"n": len(qids), "encoder": ENCODER, "tau": TAU,
               "accuracy": res, "best_single": best_m},
              open(os.path.join(HERE, "results", "kernel_analysis.json"), "w"),
              indent=2, default=float)
    print("\n  -> results/kernel_analysis.json")
    return res


if __name__ == "__main__":
    run()
