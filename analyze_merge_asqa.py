"""
ASC + optimal weights on ASQA.

WHY THIS WORKS WITHOUT A SUMMARISER
  ASC's step 4 feeds surviving facts to an LLM to compose an answer. That step
  can invent content that was in no source response -- a new failure mode with
  no analogue in selection methods.

  ASQA's metric is STR-EM: does the disambiguated short answer appear anywhere
  in the text? It is substring-based, so the CONCATENATION of surviving
  sentences scores identically to a fluent summary of them. We therefore skip
  the summariser entirely. The merged output is not readable prose, but for
  measuring interpretation coverage it is exactly equivalent -- and nothing can
  hallucinate.

WHAT IS COMPARED (identical pipeline, only the filter differs)
    best single model     one model for every question
    oracle selection      best RESPONSE per question       (needs labels)
    oracle merge          best coverage per interpretation (needs labels)
    MA-count filter       keep cluster if >= THETA MODELS contributed
                          [NOT published ASC -- see arXiv:2405.13131]
    WEIGHTED filter       keep cluster if sum(w_j) >= THETA            [ours]

  w_j is the model's mean STR-EM, 5-fold cross-fitted, NON-NEGATIVE. The
  logit(accuracy) form used elsewhere in this project goes negative whenever
  accuracy < 50% and makes a threshold filter return the empty set -- that bug
  is documented in analyze_merge.py.

Usage:  ./venv/bin/python analyze_merge_asqa.py
"""
import os, sys, json, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_asqa as R

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
TAU = 0.85          # sentence-cluster threshold
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    out = []
    for s in _SENT.split(str(text)):
        s = s.strip()
        if len(s.split()) >= 3:
            out.append(s)
    return out


def boot(a, b, n=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    r = np.random.default_rng(SEED)
    s = d[r.integers(0, len(d), (n, len(d)))].mean(axis=1) * 100
    return d.mean() * 100, np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    items = {it["qid"]: it for it in R.load_items(100000)}
    R._ITEMS.update(items)
    rows = [json.loads(l) for l in open(f"{HERE}/data/asqa_gen.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        if r["qid"] in items:
            by[r["qid"]][r["model"]] = r
    POOL = R.POOL
    qids = sorted(q for q in by if len(by[q]) == len(POOL))
    print(f"\n  ASQA  n={len(qids)} complete questions, {len(POOL)} models")

    strem = np.array([[by[q][m]["strem"] for m in POOL] for q in qids])
    best_i = int(np.argmax(strem.mean(axis=0)))
    best_vec = strem[:, best_i]
    oracle_sel = strem.max(axis=1)
    oracle_merge = np.array([R._union(by[q]) for q in qids])

    # ---- embed every sentence once
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    per_q = {}
    flat = []
    for q in qids:
        rec = []
        for m in POOL:
            for s in sentences(by[q][m]["resp"]):
                rec.append((m, s)); flat.append(s)
        per_q[q] = rec
    print(f"  {len(flat):,} sentences  ({len(flat)/max(1,len(qids)):.1f} per question)",
          flush=True)
    E = enc.encode(flat, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    off, i = {}, 0
    for q in qids:
        off[q] = (i, i + len(per_q[q])); i += len(per_q[q])

    # ---- weights: mean STR-EM, cross-fitted, non-negative
    fold = np.random.default_rng(SEED).integers(0, 5, len(qids))
    w_cv = {k: strem[fold != k].mean(axis=0) for k in range(5)}
    qi = {q: i for i, q in enumerate(qids)}

    def merge(q, wfn, theta):
        """Single-linkage cluster this question's sentences at TAU, sum support
        per cluster, keep clusters clearing theta, concatenate representatives."""
        rec = per_q[q]
        a, b = off[q]
        V = E[a:b]
        n = len(rec)
        if n == 0:
            return 0.0
        S = np.clip(V @ V.T, -1, 1)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for x in range(n):
            for y in range(x + 1, n):
                if S[x, y] >= TAU:
                    parent[find(x)] = find(y)
        sup = collections.defaultdict(float)
        seen_model = collections.defaultdict(set)
        members = collections.defaultdict(list)
        for idx, (m, s) in enumerate(rec):
            r0 = find(idx)
            members[r0].append(s)
            if m not in seen_model[r0]:          # one vote per model per cluster
                seen_model[r0].add(m)
                sup[r0] += wfn(m, q)
        keep = [max(members[r0], key=len) for r0, v in sup.items()
                if v >= theta - 1e-9]
        return R.str_em(" ".join(keep), items[q]["short_sets"])

    def sweep(wfn, thetas, label):
        best, curve = None, []
        for th in thetas:
            v = np.array([merge(q, wfn, th) for q in qids])
            curve.append((th, 100 * v.mean()))
            if best is None or v.mean() > best[1].mean():
                best = (th, v)
        print(f"\n  {label} — threshold sweep")
        for th, sc in curve:
            print(f"    theta {th:>6.3f}   STR-EM {sc:6.2f}%"
                  + ("   <- best" if th == best[0] else ""))
        return best

    cnt = sweep(lambda m, q: 1.0, [1, 2, 3, 4, 5],
                "MA-count filter (ours, ASC-style, NOT published ASC)")
    wsum = float(np.mean([w_cv[k].sum() for k in range(5)]))
    ths = np.round(np.linspace(0.15, 1.0, 10) * wsum, 3)
    wt = sweep(lambda m, q: float(w_cv[fold[qi[q]]][POOL.index(m)]), list(ths),
               "WEIGHTED filter (ours)")

    print(f"\n  {'method':<38}{'STR-EM':>9}{'vs best single':>16}")
    print("  " + "-" * 66)
    for nm, v in [(f"best single ({POOL[best_i].split('/')[-1][:16]})", best_vec),
                  (f"ASC count filter  (theta={cnt[0]})", cnt[1]),
                  (f"WEIGHTED filter   (theta={wt[0]})", wt[1]),
                  ("oracle SELECTION (needs labels)", oracle_sel),
                  ("oracle MERGE (needs labels)", oracle_merge)]:
        d = 100 * (v.mean() - best_vec.mean())
        print(f"  {nm:<38}{100*v.mean():8.2f}%{d:>+15.2f}")

    print(f"\n  the decisive comparison:")
    for lbl, a, b in [("WEIGHTED - ASC count", wt[1], cnt[1]),
                      ("WEIGHTED - best single", wt[1], best_vec),
                      ("ASC count - best single", cnt[1], best_vec)]:
        m, lo, hi = boot(a, b)
        print(f"    {lbl:<26}{m:+7.2f}   [{lo:+6.2f}, {hi:+6.2f}]"
              + ("  *" if not (lo < 0 < hi) else ""))
    print("\n  * = 95% CI excludes zero")

    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump({"n": len(qids), "tau": TAU,
               "best_single": 100*best_vec.mean(),
               "asc_count": {"theta": cnt[0], "strem": 100*cnt[1].mean()},
               "weighted": {"theta": float(wt[0]), "strem": 100*wt[1].mean()},
               "oracle_selection": 100*oracle_sel.mean(),
               "oracle_merge": 100*oracle_merge.mean()},
              open(f"{HERE}/results/asqa_merge.json", "w"), indent=2, default=float)
    print("  -> results/asqa_merge.json")


if __name__ == "__main__":
    main()
