"""
THE REAL EXPERIMENT -- aggregating actual multi-model free-form answers.

Everything before this was synthetic or a proxy. This is the test the whole
project has been building toward: 6 real models, real TriviaQA questions, real
free-form answers, graded against official alias lists.

Per question the candidate pool is the set of DISTINCT answers the models
actually produced (so K_eff varies, typically 2-6). Each aggregator must pick one.

Aggregators compared:
  MV-exact      majority vote on normalised strings                (the status quo)
  MV-cluster    cluster near-identical answers, then vote, tau tuned (Universal Self-Consistency)
  OW-L          the paper's method: accuracies from pairwise agreement, log-odds weights
  ISP           the paper's second-order method
  KWA-EM        this project: similarity kernel, beta estimated label-free by EM
  Best single   the best individual model (clairvoyant -- not a fair baseline)
  Ceiling       pick a correct answer whenever ANY model produced one

Four questions answered:
  Q1  Does KWA beat the baselines on real generations?
  Q2  Does label-free beta actually track real model skill? (never tested before)
  Q3  Is the advantage LARGER under 'natural' prompting, as the mechanism predicts?
  Q4  Do same-family models (3 Llamas) really make correlated errors?
"""
import os, json, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import kernel_agg as ka
from generate import MODELS, norm, is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TAUS = [0.55, 0.65, 0.75, 0.85, 0.92]
RESULTS = {}


def load(tag="full"):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", f"gen_{tag}.jsonl"))]
    return pd.DataFrame(rows)


def build(df, style, enc):
    """-> list of per-question dicts with pool texts, similarity matrix, correctness."""
    sub = df[df["style"] == style]
    piv = sub.pivot_table(index="qid", columns="model", values="resp", aggfunc="first")
    piv = piv.reindex(columns=MODELS).dropna()
    gold = sub.groupby("qid")["gold"].first()
    cor = sub.pivot_table(index="qid", columns="model", values="correct",
                          aggfunc="first").reindex(columns=MODELS).dropna()
    qs = []
    texts = []
    for qid, row in piv.iterrows():
        answers = [str(row[m]) for m in MODELS]
        keys = [norm(a) for a in answers]
        pool, idx = [], []
        seen = {}
        for k, a in zip(keys, answers):
            if k not in seen:
                seen[k] = len(pool); pool.append(a)
            idx.append(seen[k])
        qs.append(dict(qid=qid, answers=answers, pool=pool, idx=idx,
                       gold=gold.loc[qid],
                       agent_correct=[bool(cor.loc[qid, m]) for m in MODELS]))
        texts.extend(pool)
    E = enc(texts)
    off = 0
    for q in qs:
        n = len(q["pool"])
        V = E[off:off + n]; off += n
        q["S"] = V @ V.T
        q["pool_correct"] = [bool(c) for c in q["agent_correct"]]  # placeholder, fixed below
    return qs


def mark_pool_correct(qs, df, style):
    """Which POOL entries are correct, judged against the same alias rule."""
    sub = df[df["style"] == style]
    alias = {}
    for qid, g in sub.groupby("qid"):
        alias[qid] = g["gold"].iloc[0]
    # rebuild alias lists from the source data
    tq = pd.read_parquet(os.path.join(HERE, "data", "triviaqa_val.parquet"))
    amap = {r["question_id"]: list(r["answer"]["normalized_aliases"]) + [r["answer"]["value"]]
            for _, r in tq.iterrows()}
    for q in qs:
        al = [a for a in amap.get(q["qid"], [q["gold"]]) if a]
        q["pool_correct"] = [is_correct(p, al) for p in q["pool"]]
    return qs


# ----------------------------------------------------------------- aggregators
def mv_exact(q, rng):
    c = np.bincount(q["idx"], minlength=len(q["pool"])).astype(float)
    return int(rng.choice(np.flatnonzero(c >= c.max() - 1e-9)))

def mv_cluster(q, rng, tau):
    n = len(q["pool"]); S = q["S"]
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= tau: parent[find(i)] = find(j)
    cnt, mem = {}, {}
    for a in q["idx"]:
        r = find(a); cnt[r] = cnt.get(r, 0) + 1; mem.setdefault(r, []).append(a)
    roots = list(cnt); vals = np.array([cnt[r] for r in roots], float)
    root = roots[int(rng.choice(np.flatnonzero(vals >= vals.max() - 1e-9)))]
    m, c = np.unique(mem[root], return_counts=True)
    return int(m[int(rng.choice(np.flatnonzero(c >= c.max() - 1e-9)))])

def weighted(q, rng, w):
    sc = np.zeros(len(q["pool"]))
    for j, a in enumerate(q["idx"]): sc[a] += w[j]
    return int(rng.choice(np.flatnonzero(sc >= sc.max() - 1e-9)))

def kernel(q, rng, betas):
    sc = np.zeros(len(q["pool"]))
    for j, a in enumerate(q["idx"]): sc += betas[j] * q["S"][a]
    return int(rng.choice(np.flatnonzero(sc >= sc.max() - 1e-9)))


def pad(qs, N):
    """Ragged pools -> padded arrays for the EM estimator."""
    C = max(len(q["pool"]) for q in qs)
    A = np.zeros((len(qs), N), dtype=int)
    S = np.zeros((len(qs), C, C))
    for i, q in enumerate(qs):
        A[i] = q["idx"]
        n = len(q["pool"]); S[i, :n, :n] = q["S"]
    return A, S


def agreement_accuracies(qs, N, K_eff):
    """The paper's OW-L, on exact-match agreement."""
    ag = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                ag[i, j] = np.mean([q["idx"][i] == q["idx"][j] for q in qs])
    try:
        return ka.estimate_acc_from_agreement(ag, K_eff=max(2, int(round(K_eff))))
    except Exception:
        return np.full(N, 1.0 / max(2, K_eff) + 0.05)


def isp_pick(qs, N, rng_seed):
    """The paper's ISP: votes + agreement bonus, on exact-match identity."""
    C = max(len(q["pool"]) for q in qs)
    # second-order table on 'did agents i and j give the same answer'
    same = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j: same[i, j] = np.mean([q["idx"][i] == q["idx"][j] for q in qs])
    picks = []
    rng = np.random.default_rng(rng_seed)
    for q in qs:
        n = len(q["pool"])
        votes = np.bincount(q["idx"], minlength=n).astype(float)
        bonus = np.zeros(n)
        for s in range(n):
            for i in range(N):
                vals = [same[i, j] if q["idx"][j] == s else (1 - same[i, j]) / max(1, n - 1)
                        for j in range(N) if j != i]
                bonus[s] += np.mean(vals)
        Keff = max(2, n)
        sc = votes + bonus / (Keff - 1)
        picks.append(int(rng.choice(np.flatnonzero(sc >= sc.max() - 1e-9))))
    return picks


def evaluate(qs, style, enc_name):
    N = len(MODELS)
    K_eff = np.mean([len(q["pool"]) for q in qs])
    acc_true = np.array([np.mean([q["agent_correct"][j] for q in qs]) for j in range(N)])
    A, S = pad(qs, N)
    beta_hat, _ = ka.em_estimate_beta(A, S, support="observed", n_iter=40)

    x_owl = agreement_accuracies(qs, N, K_eff)
    Kw = max(2, K_eff)
    w_owl = np.log((Kw - 1) * np.clip(x_owl, 1e-6, 1 - 1e-6) / (1 - np.clip(x_owl, 1e-6, 1 - 1e-6)))
    w_orc = np.log((Kw - 1) * np.clip(acc_true, 1e-6, 1 - 1e-6) / (1 - np.clip(acc_true, 1e-6, 1 - 1e-6)))

    def score(fn):
        rng = np.random.default_rng(0)
        return 100 * np.mean([q["pool_correct"][fn(q, rng)] for q in qs])

    res = {}
    res["MV-exact"] = score(mv_exact)
    best_tau, best = None, -1
    for tau in TAUS:
        v = score(lambda q, r, t=tau: mv_cluster(q, r, t))
        if v > best: best, best_tau = v, tau
    res["MV-cluster"] = best; res["_tau"] = best_tau
    res["OW-L"] = score(lambda q, r: weighted(q, r, w_owl))
    res["OW-oracle"] = score(lambda q, r: weighted(q, r, w_orc))
    picks = isp_pick(qs, N, 0)
    res["ISP"] = 100 * np.mean([q["pool_correct"][p] for q, p in zip(qs, picks)])
    res["KWA-EM"] = score(lambda q, r: kernel(q, r, beta_hat))
    res["BestSingle"] = 100 * acc_true.max()
    res["Ceiling"] = 100 * np.mean([any(q["pool_correct"]) for q in qs])
    res["_beta_hat"] = beta_hat.tolist()
    res["_acc_true"] = acc_true.tolist()
    res["_x_owl"] = x_owl.tolist()
    res["_K_eff"] = float(K_eff)
    res["_n"] = len(qs)
    return res


def main(tag="full"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    enc = lambda t: m.encode(list(t), batch_size=256, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    df = load(tag)
    print(f"  loaded {len(df):,} responses\n")
    for style in ("terse", "natural"):
        qs = build(df, style, enc)
        qs = mark_pool_correct(qs, df, style)
        r = evaluate(qs, style, "MiniLM")
        RESULTS[style] = r
        print(f"\n{'='*78}\n  {style.upper()}   n={r['_n']}   mean distinct answers/question = {r['_K_eff']:.2f}\n{'='*78}")
        order = ["MV-exact", "MV-cluster", "ISP", "OW-L", "KWA-EM", "OW-oracle", "BestSingle", "Ceiling"]
        base = r["MV-exact"]
        print(f"  {'method':<14}{'accuracy':>10}{'vs MV-exact':>14}")
        print("  " + "-" * 38)
        for k in order:
            print(f"  {k:<14}{r[k]:>10.2f}{r[k]-base:>+14.2f}")
        b, a = np.array(r["_beta_hat"]), np.array(r["_acc_true"])
        rho = spearmanr(b, a).statistic
        print(f"\n  LABEL-FREE SKILL ESTIMATION (the core claim, never tested on real models):")
        print(f"    {'model':<38}{'true acc':>10}{'beta_hat':>10}")
        for j, mm in enumerate(MODELS):
            print(f"    {mm:<38}{100*a[j]:>9.1f}%{b[j]:>10.2f}")
        print(f"    Spearman rank correlation: {rho:+.3f}")
        RESULTS[style]["_spearman"] = float(rho)
    json.dump(RESULTS, open(os.path.join(OUT, f"real_{tag}.json"), "w"), indent=2)
    print(f"\n  -> results/real_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
