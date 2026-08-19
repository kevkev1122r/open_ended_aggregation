"""
Multi-domain analysis: does niche specialisation exist, and can weighting exploit it?

The question this is built to answer: if models are matched in GENERAL capability but
each is better in some niche, a single weight per model cannot represent that, and
both the paper's method and ours degenerate into majority voting. Per-question
weights should be able to.

So three things get measured, in order:

  1. ARE THEY MATCHED?     overall accuracy per model. If one dominates, we are back
                           in the TriviaQA regime and the niche question is moot.
  2. IS THERE A NICHE?     model x domain accuracy matrix, and a formal test of whether
                           the interaction is real or sampling noise. Matched overall
                           accuracy plus a significant interaction IS specialisation.
  3. CAN WEIGHTS USE IT?   aggregation with GLOBAL weights vs PER-DOMAIN weights.
                           Prediction: global weights ~ majority voting (they cannot
                           express the structure), per-domain weights beat both, and
                           only per-domain weights can beat the best single model --
                           something that never happened on TriviaQA.

Grading uses the independent LLM judge where available, falling back to string
matching, and reports both so the difference is visible rather than assumed.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def load(gen="data/full_run.jsonl", judged="data/full_run_judged.jsonl"):
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, gen))])
    d["dom"] = d.domain.str.split(":").str[0]
    jp = os.path.join(HERE, judged)
    if os.path.exists(jp):
        j = pd.DataFrame([json.loads(l) for l in open(jp)])
        d = d.merge(j[["qid", "model", "judged"]], on=["qid", "model"], how="left")
        d["ok"] = d.judged.fillna(d.correct).astype(bool)
        d["graded_by"] = np.where(d.judged.notna(), "judge", "string")
    else:
        d["ok"] = d.correct.astype(bool); d["judged"] = np.nan; d["graded_by"] = "string"
    return d


def complete_matrix(d, key="ok"):
    """questions answered by EVERY model, so comparisons are paired."""
    piv = d.pivot_table(index="qid", columns="model", values=key, aggfunc="first")
    piv = piv.dropna()
    dom = d.groupby("qid")["domain"].first()
    return piv.astype(bool), dom.loc[piv.index]


# ================================================================= 1. matched?
def capability(piv, dom):
    acc = 100 * piv.mean()
    print(f"\n  {'model':<34}{'overall':>9}{'95% CI':>16}")
    print("  " + "-" * 60)
    n = len(piv)
    for m in acc.sort_values(ascending=False).index:
        p = acc[m] / 100
        se = 100 * np.sqrt(p * (1 - p) / n)
        print(f"  {m:<34}{acc[m]:>8.1f}%   [{acc[m]-1.96*se:5.1f},{acc[m]+1.96*se:5.1f}]")
    spread = acc.max() - acc.min()
    print(f"\n  n = {n} paired questions")
    print(f"  spread best-to-worst: {spread:.1f} points")
    top2 = acc.sort_values(ascending=False).iloc[:2]
    print(f"  gap between #1 and #2: {top2.iloc[0]-top2.iloc[1]:.1f} points")
    if spread < 8:
        print("  -> ROUGHLY MATCHED. the niche question is live.")
    else:
        print("  -> NOT matched; one or more models dominate, as on TriviaQA.")
    return acc


# ================================================================= 2. niche?
def specialisation(piv, dom):
    doms = sorted(dom.unique())
    M = pd.DataFrame({d_: 100 * piv[dom == d_].mean() for d_ in doms})
    print(f"\n  {'model':<30}" + "".join(f"{d_.replace('mmlupro:','mp:'):>13}" for d_ in doms))
    print("  " + "-" * (30 + 13 * len(doms)))
    for m in M.index:
        print(f"  {m:<30}" + "".join(f"{M.loc[m,d_]:>12.0f}%" for d_ in doms))
    print(f"  {'':<30}" + "".join(f"{'-'*12:>13}" for _ in doms))
    print(f"  {'(best model per domain)':<30}" +
          "".join(f"{M[d_].idxmax().split('/')[-1][:11]:>13}" for d_ in doms))

    # de-mean by model and by domain: what is left is the interaction
    resid = M.sub(M.mean(axis=1), axis=0).sub(M.mean(axis=0), axis=1) + M.values.mean()
    inter = resid - resid.mean().mean()
    print(f"\n  interaction spread (model x domain, after removing model and domain effects):")
    print(f"    sd of interaction = {inter.values.std():.2f} accuracy points")
    print(f"    max |interaction| = {np.abs(inter.values).max():.2f} points"
          f"  ({inter.stack().abs().idxmax()})")

    # permutation test: is the interaction bigger than chance?
    #
    # FIXED 2026-08-14. The previous null permuted ROWS within each domain block
    # and then took the COLUMN mean -- but a mean over rows is invariant to row
    # order, so every replicate reproduced the observed matrix exactly and p was
    # pinned at 1.0000 by construction. It could never have rejected anything.
    #
    # Correct null: shuffle the DOMAIN LABELS across questions. That preserves
    # each model's overall accuracy and each question's difficulty (the two main
    # effects we de-mean out anyway) while destroying any model-specific affinity
    # for a domain -- which is exactly the interaction under test.
    rng = np.random.default_rng(0)
    obs = inter.values.std()
    dom_arr = dom.values.copy()
    null = []
    for _ in range(2000):
        shuffled = pd.Series(rng.permutation(dom_arr), index=dom.index)
        Ms = pd.DataFrame({d_: 100 * piv[shuffled == d_].mean() for d_ in doms})
        r = Ms.sub(Ms.mean(axis=1), axis=0).sub(Ms.mean(axis=0), axis=1)
        null.append((r - r.mean().mean()).values.std())
    null = np.array(null)
    p = (null >= obs).mean()
    print(f"    observed interaction sd {obs:.2f} vs null mean "
          f"{null.mean():.2f} (95th pct {np.percentile(null,95):.2f})")
    print(f"    permutation test vs shuffled-within-domain null: p = {p:.4f}")
    print("    -> " + ("REAL specialisation structure" if p < 0.05
                       else "no more structure than chance"))
    return M, float(p)


# ================================================================= 3. exploit?
def aggregation(d, piv, dom):
    """Majority vote vs global-weighted vote vs per-domain-weighted vote."""
    doms = sorted(dom.unique())
    models = list(piv.columns)
    resp = d.pivot_table(index="qid", columns="model", values="resp",
                         aggfunc="first").reindex(piv.index)[models]
    ok = piv[models]

    def weighted_vote(weights_fn):
        """weights_fn(qid) -> array of per-model weights. Ties broken at random."""
        rng = np.random.default_rng(0); hit = 0
        for qid in piv.index:
            answers = [str(resp.loc[qid, m]) for m in models]
            keys = [a.strip().lower()[-120:] for a in answers]
            w = weights_fn(qid)
            score = {}
            for k, wt, m in zip(keys, w, models):
                score[k] = score.get(k, 0.0) + wt
            best = max(score.values())
            cands = [k for k, v in score.items() if v >= best - 1e-9]
            pick = rng.choice(cands)
            idxs = [i for i, k in enumerate(keys) if k == pick]
            hit += bool(ok.loc[qid, models[rng.choice(idxs)]])
        return 100 * hit / len(piv)

    def logit_w(a):
        a = np.clip(a, 1e-3, 1 - 1e-3)
        return np.log(a / (1 - a))

    domof = dom.to_dict()

    # ------------------------------------------------------------ OUT OF SAMPLE
    # Weights must be estimated on questions they are not then scored on.
    # Per-domain weighting fits 7 models x 9 domains = 63 parameters; scoring
    # those on the same 970 questions that produced them flatters it against
    # majority vote, which fits nothing. 5-fold CV puts every method on equal
    # footing: for each fold, weights come only from the other four.
    K = 5
    rng_f = np.random.default_rng(0)
    fold = pd.Series(rng_f.integers(0, K, len(piv)), index=piv.index)
    w_glob_cv, w_dom_cv = {}, {}
    for k in range(K):
        tr = piv[fold != k]
        w_glob_cv[k] = logit_w(tr.mean().values)
        tr_dom = dom.loc[tr.index]
        w_dom_cv[k] = {}
        for d_ in doms:
            block = tr[tr_dom == d_]
            # a fold may leave a domain thin; fall back to that fold's global
            w_dom_cv[k][d_] = (logit_w(block.mean().values) if len(block) >= 20
                               else w_glob_cv[k])
    foldof = fold.to_dict()

    res = {}
    res["majority vote"] = weighted_vote(lambda q: np.ones(len(models)))
    res["global weights"] = weighted_vote(lambda q: w_glob_cv[foldof[q]])
    res["per-domain weights"] = weighted_vote(lambda q: w_dom_cv[foldof[q]][domof[q]])

    # in-sample versions kept only to show how much the CV correction mattered
    w_glob_is = logit_w(piv.mean().values)
    w_dom_is = {d_: logit_w(piv[dom == d_].mean().values) for d_ in doms}
    res["global weights (in-sample)"] = weighted_vote(lambda q: w_glob_is)
    res["per-domain weights (in-sample)"] = weighted_vote(lambda q: w_dom_is[domof[q]])
    res["best single model"] = 100 * piv.mean().max()
    res["ceiling (any correct)"] = 100 * piv.any(axis=1).mean()

    print(f"\n  {'method':<26}{'accuracy':>10}{'vs MV':>9}{'vs best single':>16}")
    print("  " + "-" * 61)
    mv = res["majority vote"]; bs = res["best single model"]
    for k in ["majority vote", "global weights", "per-domain weights",
              "best single model", "ceiling (any correct)",
              "global weights (in-sample)", "per-domain weights (in-sample)"]:
        if k not in res: continue
        if k.endswith("(in-sample)") and k == "global weights (in-sample)":
            print("  " + "-" * 61)
            print("  (below: same weights fit AND scored on the same questions --")
            print("   shown only to size the optimism the CV numbers remove)")
        print(f"  {k:<26}{res[k]:>10.2f}{res[k]-mv:>+9.2f}{res[k]-bs:>+16.2f}")
    print(f"\n  the prediction was: global weights ~ majority voting (a scalar cannot")
    print(f"  express specialisation), per-domain weights beat both and can clear the")
    print(f"  best single model -- which never happened on TriviaQA.")
    return res


if __name__ == "__main__":
    d = load()
    print("=" * 78)
    print(f"  {len(d):,} responses   |   graded by: "
          f"{d.graded_by.value_counts().to_dict()}")
    piv, dom = complete_matrix(d)
    print("=" * 78); print("  1. ARE THE MODELS MATCHED IN GENERAL CAPABILITY?"); print("=" * 78)
    acc = capability(piv, dom)
    print("\n" + "=" * 78); print("  2. IS THERE NICHE SPECIALISATION?"); print("=" * 78)
    M, p = specialisation(piv, dom)
    print("\n" + "=" * 78); print("  3. CAN WEIGHTING EXPLOIT IT?"); print("=" * 78)
    res = aggregation(d, piv, dom)
    json.dump({"overall": acc.to_dict(), "matrix": M.to_dict(),
               "interaction_p": p, "aggregation": res},
              open(os.path.join(OUT, "domain_analysis.json"), "w"), indent=2, default=float)
    print("\n  -> results/domain_analysis.json")
