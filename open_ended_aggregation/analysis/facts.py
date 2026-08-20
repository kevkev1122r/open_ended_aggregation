"""
Aggregation comparison on FACTS Grounding — the first run where the method's
preconditions all hold at once.

WHY THIS RUN IS DIFFERENT FROM v2
    answers            6 words  ->  214 words
    contested          32.7%    ->  61.1%
    headroom            9.8 pts ->  19.4 pts
    grading            match a reference string -> is every sentence grounded
                                                   in the provided document

  In v2 the kernel was idle: 62% of answers were <=5 words with a canonical form,
  where exact matching is already the correct similarity function and anything
  softer can only add noise. Here no canonical form exists, so exact matching has
  to fail and the kernel finally has something to do.

  The aggregators are imported from verify_aggregation, which unit-tests each one
  against a hand-computed case before use. That file exists because three of the
  aggregation bugs in the previous run were mine, all flattering the method.

WHAT TO EXPECT, WRITTEN DOWN BEFORE LOOKING
  Exact matching should collapse to near-random, because two 214-word answers
  essentially never normalise to the same string. If the kernel does NOT beat it
  clearly here, the method does not work -- this is the regime it was designed
  for and there is no remaining excuse about benchmark shape.

Usage:  ./venv/bin/python analyze_facts.py
"""
import os, sys, json, collections
import numpy as np
import verify_aggregation as V          # unit-tested aggregators + normaliser
from open_ended_aggregation.methods import kernel as K

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
POOL = ["grok-4.3", "Kimi-K2.5", "Cohere-command-a-plus-05-2026",
        "MAI-Thinking-1", "DeepSeek-V4-Flash"]
ENCODER = "sentence-transformers/all-mpnet-base-v2"
SEED = 0


def load():
    gen = [json.loads(l) for l in open(f"{HERE}/data/facts_gen.jsonl")]
    jud = {(r["qid"], r["model"]): r for r in
           (json.loads(l) for l in open(f"{HERE}/data/facts_judged.jsonl"))}
    by = collections.defaultdict(dict)
    for r in gen:
        k = (r["qid"], r["model"])
        if k in jud:
            by[r["qid"]][r["model"]] = {"resp": r["resp"],
                                        "ok": bool(jud[k]["judged"]),
                                        "grounded": jud[k].get("grounded")}
    qids = sorted(q for q in by if len(by[q]) == len(POOL))
    return by, qids


def main():
    by, qids = load()
    N = len(POOL)
    if len(qids) < 30:
        print(f"  only {len(qids)} complete items — run more first"); return
    acc = np.array([[by[q][m]["ok"] for m in POOL] for q in qids], float)
    print(f"\n  {len(qids)} items answered and graded by all {N} models")
    print(f"  mean response {np.mean([len(by[q][m]['resp'].split()) for q in qids for m in POOL]):.0f} words")

    # ---------- how often does exact matching even fire?
    keys = []
    for q in qids:
        ks = [V.key(V.clean(by[q][m]["resp"])) for m in POOL]
        keys.append(ks)
    alld = 100*np.mean([len(set(k)) == N for k in keys])
    print(f"  all {N} answers distinct after normalising: {alld:.1f}%"
          f"   <- exact matching is near-blind above ~90%")

    # ---------- embed
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    texts, uniq = [], []
    for q, ks in zip(qids, keys):
        u = list(dict.fromkeys(ks))
        rep = {}
        for m, k in zip(POOL, ks):
            rep.setdefault(k, V.clean(by[q][m]["resp"])[:2000] or "(no answer)")
        uniq.append((ks, u)); texts += [rep[k] for k in u]
    print(f"  embedding {len(texts)} responses …", flush=True)
    E = enc.encode(texts, batch_size=64, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    off = np.cumsum([0] + [len(u) for _, u in uniq])
    S = [np.clip(E[off[i]:off[i+1]] @ E[off[i]:off[i+1]].T, -1, 1) for i in range(len(qids))]

    # ---------- weights: supervised (5-fold CV) and label-free (EM)
    def logit(a):
        a = np.clip(a, 1e-3, 1-1e-3); return np.log(a/(1-a))
    fold = np.random.default_rng(SEED).integers(0, 5, len(qids))
    w_cv = {k: logit(acc[fold != k].mean(axis=0)) for k in range(5)}
    Amax = max(len(u) for _, u in uniq)
    A = np.zeros((len(qids), N), int); Sbig = np.zeros((len(qids), Amax, Amax))
    for i, (ks, u) in enumerate(uniq):
        pos = {k: j for j, k in enumerate(u)}
        A[i] = [pos[k] for k in ks]
        Sbig[i, :len(u), :len(u)] = S[i]
    print("  running label-free EM …", flush=True)
    beta_em, _ = K.em_estimate_beta(A, Sbig, n_iter=30, support="observed")
    from scipy import stats
    sp = stats.spearmanr(beta_em, acc.mean(axis=0))
    print(f"  EM beta vs true accuracy: Spearman {sp.statistic:+.3f}  p={sp.pvalue:.3f}")
    for m, b, a in sorted(zip(POOL, beta_em, 100*acc.mean(axis=0)), key=lambda t: -t[1]):
        print(f"      {m:<34}beta {b:7.3f}   acc {a:5.1f}%")

    # ---------- score
    def run(fn):
        h = []
        for i, q in enumerate(qids):
            ks, _ = uniq[i]
            pick = fn(i, ks, S[i])
            win = [m for m, k in zip(POOL, ks) if k == pick]
            h.append(by[q][np.random.default_rng(i).choice(win)]["ok"])
        return np.array(h, bool)

    res = {
        "majority vote":        run(lambda i, ks, s: V.agg_majority(ks, s, np.random.default_rng(i))),
        "medoid / cluster":     run(lambda i, ks, s: V.agg_cluster(ks, s, np.random.default_rng(i))),
        "OW exact (weights)":   run(lambda i, ks, s: V.agg_majority(ks, s, np.random.default_rng(i), w=w_cv[fold[i]])),
        "KWA supervised beta":  run(lambda i, ks, s: V.agg_kernel(ks, s, np.random.default_rng(i), w=w_cv[fold[i]])),
        "KWA label-free beta":  run(lambda i, ks, s: V.agg_kernel(ks, s, np.random.default_rng(i), w=beta_em)),
    }
    per = 100*acc.mean(axis=0); bi = int(np.argmax(per))
    bsv = acc[:, bi].astype(bool)

    def boot(a, b, n=10000):
        d = a.astype(float) - b.astype(float)
        r = np.random.default_rng(SEED)
        s_ = d[r.integers(0, len(d), (n, len(d)))].mean(axis=1)*100
        return d.mean()*100, np.percentile(s_, 2.5), np.percentile(s_, 97.5)

    print(f"\n  {'method':<26}{'acc':>8}{'vs best':>10}{'95% CI':>20}")
    print("  " + "-"*66)
    for k, v in res.items():
        m, lo, hi = boot(v, bsv)
        print(f"  {k:<26}{100*v.mean():>8.2f}{m:>+10.2f}   [{lo:>+6.2f},{hi:>+6.2f}]"
              + ("  *" if not (lo < 0 < hi) else ""))
    print("  " + "-"*66)
    print(f"  {'best single model':<26}{per[bi]:>8.2f}   ({POOL[bi]})")
    print(f"  {'ceiling (any correct)':<26}{100*(acc.sum(axis=1) > 0).mean():>8.2f}")
    print(f"  {'mean model accuracy':<26}{per.mean():>8.2f}   <- a random pick")

    print(f"\n  the decisive comparison, kernel vs exact matching on long answers:")
    m, lo, hi = boot(res["KWA supervised beta"], res["OW exact (weights)"])
    print(f"    KWA - OW exact   {m:+.2f}   [{lo:+.2f}, {hi:+.2f}]"
          + ("  * significant" if not (lo < 0 < hi) else "  (spans zero)"))

    json.dump({"n": len(qids), "all_distinct_pct": alld,
               "accuracy": {k: 100*v.mean() for k, v in res.items()},
               "best_single": per[bi], "best_model": POOL[bi],
               "ceiling": 100*(acc.sum(axis=1) > 0).mean(),
               "em_spearman": float(sp.statistic), "em_beta": beta_em.tolist()},
              open(f"{HERE}/results/facts_analysis.json", "w"), indent=2, default=float)
    print("\n  -> results/facts_analysis.json")


if __name__ == "__main__":
    main()
