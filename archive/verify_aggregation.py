"""
Independent re-derivation of the aggregation table, written to be checkable.

Deliberately does NOT import analyze_domains or analyze_kernel. It reads the
JSONL directly and reimplements every aggregator, so a disagreement with those
files is informative rather than tautological. kernel_agg is imported for
em_estimate_beta only, because that IS the method under test.

Three things run before any result is printed:

  1. UNIT TESTS on hand-computed cases. Each aggregator is checked against an
     input whose correct output can be worked out on paper. An aggregator that
     cannot reproduce a 3-2 majority has no business reporting an accuracy.
  2. A FALSE-MERGE AUDIT of the normaliser. Deduplication is only sound if
     answers it merges really are the same answer. The judge graded each
     response independently, so if two merged answers were graded differently,
     the merge is wrong. That is a direct empirical test, not an opinion.
  3. A DEGENERACY CHECK: how often each aggregator is reduced to a coin flip.
     An aggregator that ties on most questions is reporting the mean model
     accuracy while looking like a method.

Usage:  ./venv/bin/python verify_aggregation.py
"""
import os, re, json, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0

# ----------------------------------------------------------------- normalising
CTRL  = re.compile(r"<\|[^|]*\|>")
LATEX = re.compile(r"\\[()\[\]]|\$\$?")
LEAD  = re.compile(r"(?i)^\s*(the\s+)?(final\s+)?answer\s*(is)?\s*[:\-]?\s*")
ART   = re.compile(r"\b(a|an|the)\b", re.I)
PUNCT = re.compile(r"[^a-z0-9 ]")


def final_line(resp):
    """Last non-empty line -- models are prompted to end with the answer."""
    lines = [l.strip() for l in str(resp).strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def clean(t):
    t = CTRL.sub("", t)
    t = t.replace("**", "").replace("__", "")
    t = LATEX.sub("", t)
    t = LEAD.sub("", t)
    t = re.sub(r"[\s.;,]+$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def key(t):
    """Merge key. Conservative on purpose: case, punctuation and articles only."""
    s = clean(t).lower().replace("\u2019", "'")
    s = PUNCT.sub(" ", s)
    s = ART.sub(" ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------- aggregators
def agg_majority(keys, S, rng, w=None):
    """Count votes on merged answers. w=None -> unweighted."""
    w = np.ones(len(keys)) if w is None else w
    score = collections.defaultdict(float)
    for k, wt in zip(keys, w):
        score[k] += wt
    best = max(score.values())
    return rng.choice([k for k, v in score.items() if v >= best - 1e-9])


def agg_cluster(keys, S, rng, tau=0.90, w=None):
    """Single-linkage cluster at tau, pick the heaviest cluster, then its
    heaviest member."""
    w = np.ones(len(keys)) if w is None else w
    uniq = list(dict.fromkeys(keys))
    pos = {k: i for i, k in enumerate(uniq)}
    parent = list(range(len(uniq)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if S[i, j] >= tau:
                parent[find(i)] = find(j)
    cw = collections.defaultdict(float); members = collections.defaultdict(list)
    for k, wt in zip(keys, w):
        r = find(pos[k]); cw[r] += wt; members[r].append((k, wt))
    best = max(cw.values())
    root = rng.choice([r for r, v in cw.items() if v >= best - 1e-9])
    mw = collections.defaultdict(float)
    for k, wt in members[root]:
        mw[k] += wt
    bm = max(mw.values())
    return rng.choice([k for k, v in mw.items() if v >= bm - 1e-9])


def agg_kernel(keys, S, rng, w):
    """argmax_s sum_j w_j * sim(a_j, s), s restricted to observed answers."""
    uniq = list(dict.fromkeys(keys))
    pos = {k: i for i, k in enumerate(uniq)}
    sc = np.zeros(len(uniq))
    for k, wt in zip(keys, w):
        sc += wt * S[pos[k]]
    best = sc.max()
    return uniq[int(rng.choice(np.flatnonzero(sc >= best - 1e-9)))]


# ---------------------------------------------------------------- unit tests
def unit_tests():
    rng = np.random.default_rng(0)
    I = np.eye(4)                      # orthogonal: no answer resembles another
    ok = True

    # 3-2 majority must win, unweighted
    got = agg_majority(["a","a","a","b","b"], I, rng)
    ok &= (got == "a"); print(f"    majority 3-2                -> {got!r}  {'ok' if got=='a' else 'FAIL'}")

    # 2-3 minority wins if its supporters carry enough weight
    got = agg_majority(["a","a","a","b","b"], I, rng, w=np.array([1,1,1,5,5.]))
    ok &= (got == "b"); print(f"    weighted 3-2 overturned     -> {got!r}  {'ok' if got=='b' else 'FAIL'}")

    # clustering must merge near-duplicates: 2 'a' + 2 'a-like' beats 3 'b'
    S = np.array([[1.0, 0.95, 0.0],
                  [0.95, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    got = agg_cluster(["a","a2","a","a2","b","b","b"], S, rng, tau=0.90)
    ok &= (got in ("a","a2")); print(f"    cluster 4(merged) vs 3 b    -> {got!r}  {'ok' if got in ('a','a2') else 'FAIL'}")

    # kernel with equal weights picks the answer nearest everything else
    S2 = np.array([[1.0, 0.9, 0.9],
                   [0.9, 1.0, 0.2],
                   [0.9, 0.2, 1.0]])
    got = agg_kernel(["m","x","y"], S2, rng, w=np.ones(3))
    ok &= (got == "m"); print(f"    kernel picks the medoid     -> {got!r}  {'ok' if got=='m' else 'FAIL'}")

    # normaliser
    pairs = [("1 inch<|END_TEXT|>", "1 inch", True),
             ("**$75.**", "$75", True),
             ("Answer: Tina Turner", "tina turner", True),
             ("5 mm", "5 cm", False),
             ("Paris", "Lyon", False)]
    for a, b, same in pairs:
        got = (key(a) == key(b))
        ok &= (got == same)
        print(f"    norm {a[:22]!r} vs {b[:14]!r} -> {'merge' if got else 'distinct'}"
              f"  {'ok' if got==same else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- the data
def load():
    gen = [json.loads(l) for l in open(f"{HERE}/data/v2.jsonl")]
    jud = {}
    jp = f"{HERE}/data/v2_judged.jsonl"
    if os.path.exists(jp):
        for l in open(jp):
            r = json.loads(l); jud[(r["qid"], r["model"])] = r["judged"]
    rows = []
    for r in gen:
        k = (r["qid"], r["model"])
        rows.append({**r, "ok": bool(jud.get(k, r["correct"])),
                     "graded": "judge" if k in jud else "string"})
    return rows


def main():
    print("\n  1. UNIT TESTS")
    if not unit_tests():
        print("\n  !! a unit test failed -- results below are not trustworthy")
        sys.exit(1)

    rows = load()
    models = sorted({r["model"] for r in rows})
    byq = collections.defaultdict(dict)
    for r in rows:
        byq[r["qid"]][r["model"]] = r
    qids = [q for q, v in byq.items() if len(v) == len(models)]
    N = len(models)
    print(f"\n  {len(rows):,} rows, {len(models)} models, "
          f"{len(qids):,} questions answered by all of them")

    # ---- 2. false-merge audit -------------------------------------------
    print("\n  2. FALSE-MERGE AUDIT  (two answers merged but graded differently)")
    merged = bad = 0
    examples = []
    for q in qids:
        groups = collections.defaultdict(list)
        for m in models:
            groups[key(final_line(byq[q][m]["resp"]))].append(byq[q][m])
        for k, g in groups.items():
            if len(g) < 2: continue
            merged += 1
            if len({x["ok"] for x in g}) > 1:
                bad += 1
                if len(examples) < 4:
                    examples.append((q, [(final_line(x["resp"])[:46], x["ok"]) for x in g]))
    print(f"    merge groups of size>=2: {merged:,}")
    print(f"    groups whose members were graded DIFFERENTLY: {bad}  "
          f"({100*bad/max(merged,1):.2f}%)")
    for q, ex in examples:
        print(f"      qid {q}: " + " | ".join(f"{t!r}->{o}" for t, o in ex))
    if bad / max(merged, 1) > 0.02:
        print("    !! normaliser is merging answers the judge treats as different")

    # ---- 3. embeddings ---------------------------------------------------
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    uniq_per_q, flat = [], []
    for q in qids:
        ks = [key(final_line(byq[q][m]["resp"])) for m in models]
        u = list(dict.fromkeys(ks))
        uniq_per_q.append((ks, u))
        flat += [clean(final_line(byq[q][m]["resp"])) or "(no answer)" for m in models]
    texts = []
    for (ks, u), q in zip(uniq_per_q, qids):
        rep = {}
        for m, k in zip(models, ks):
            rep.setdefault(k, clean(final_line(byq[q][m]["resp"])) or "(no answer)")
        texts += [rep[k] for k in u]
    V = enc.encode(texts, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    off = np.cumsum([0] + [len(u) for _, u in uniq_per_q])
    Smats = [np.clip(V[off[i]:off[i+1]] @ V[off[i]:off[i+1]].T, -1, 1)
             for i in range(len(qids))]

    # ---- 4. weights ------------------------------------------------------
    acc = np.array([[byq[q][m]["ok"] for m in models] for q in qids], float)
    def logit(a):
        a = np.clip(a, 1e-3, 1 - 1e-3); return np.log(a / (1 - a))
    rngf = np.random.default_rng(SEED)
    fold = rngf.integers(0, 5, len(qids))
    w_cv = {k: logit(acc[fold != k].mean(axis=0)) for k in range(5)}

    import kernel_agg as K
    Amax = max(len(u) for _, u in uniq_per_q)
    A = np.zeros((len(qids), N), int); Sbig = np.zeros((len(qids), Amax, Amax))
    for i, (ks, u) in enumerate(uniq_per_q):
        pos = {k: j for j, k in enumerate(u)}
        A[i] = [pos[k] for k in ks]
        Sbig[i, :len(u), :len(u)] = Smats[i]
    beta_em, _ = K.em_estimate_beta(A, Sbig, n_iter=30, support="observed")

    # ---- 5. score --------------------------------------------------------
    def score(fn):
        rng = np.random.default_rng(SEED); hits = []; ties = 0
        for i, q in enumerate(qids):
            ks, u = uniq_per_q[i]
            pick = fn(i, q, ks, Smats[i])
            winners = [m for m, k in zip(models, ks) if k == pick]
            ties += (len(set(ks)) == N)
            hits.append(byq[q][rng.choice(winners)]["ok"])
        return np.array(hits, bool), ties

    res = {}
    res["majority vote"]        = score(lambda i,q,ks,S: agg_majority(ks, S, np.random.default_rng(i)))
    res["medoid / cluster"]     = score(lambda i,q,ks,S: agg_cluster(ks, S, np.random.default_rng(i)))
    res["OW exact (weights)"]   = score(lambda i,q,ks,S: agg_majority(ks, S, np.random.default_rng(i), w=w_cv[fold[i]]))
    res["KWA supervised beta"]  = score(lambda i,q,ks,S: agg_kernel(ks, S, np.random.default_rng(i), w=w_cv[fold[i]]))
    res["KWA label-free beta"]  = score(lambda i,q,ks,S: agg_kernel(ks, S, np.random.default_rng(i), w=beta_em))

    per_model = acc.mean(axis=0) * 100
    best_i = int(np.argmax(per_model))
    bsv = acc[:, best_i].astype(bool)
    ceiling = 100 * (acc.sum(axis=1) > 0).mean()

    def boot(a, b, n=10000):
        df = a.astype(float) - b.astype(float)
        r = np.random.default_rng(SEED)
        s = df[r.integers(0, len(df), (n, len(df)))].mean(axis=1) * 100
        return df.mean() * 100, np.percentile(s, 2.5), np.percentile(s, 97.5)

    print("\n  3. AGGREGATION TABLE  (independent re-derivation)")
    print(f"    {'method':<26}{'acc':>8}{'vs best':>10}{'95% CI':>19}{'all-tied':>10}")
    print("    " + "-" * 73)
    for k, (v, ties) in res.items():
        m, lo, hi = boot(v, bsv)
        star = "" if lo < 0 < hi else " *"
        print(f"    {k:<26}{100*v.mean():>8.2f}{m:>+10.2f}   [{lo:>+6.2f},{hi:>+6.2f}]"
              f"{100*ties/len(qids):>9.1f}%{star}")
    print("    " + "-" * 73)
    print(f"    {'best single model':<26}{per_model[best_i]:>8.2f}   ({models[best_i]})")
    print(f"    {'ceiling (any correct)':<26}{ceiling:>8.2f}")
    print(f"    {'mean model accuracy':<26}{per_model.mean():>8.2f}"
          f"   <- what a random pick scores")


if __name__ == "__main__":
    main()
