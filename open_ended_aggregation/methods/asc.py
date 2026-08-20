"""
ASC exactly as published (arXiv:2405.13131, EMNLP 2024), with ONE substitution:
the m stochastic samples from a single model are replaced by the responses of
multiple agents. Nothing else is changed.

PAPER PIPELINE, QAMPARI "list mode"  (paper section: list datasets)
  1. atomic units : each list item used directly -- no sentence splitter
  2. clustering   : surface-form; two units share a cluster if their NORMALISED
                    EDIT DISTANCE is below a threshold
  3. representative: the FIRST item of each cluster
  4. strength     : the COUNT OF CLUSTER MEMBERS
                    (NOT deduplicated per agent -- the paper counts members, and
                     with m=50 samples from one model that is what it means. We
                     do not "fix" this; substituting agents for samples is the
                     only permitted change.)
  5. filtering    : drop every cluster with count < Theta
  6. composition  : surviving representatives are fed back to an LLM with the
                    paper's P_combine prompt to produce the final answer
  7. Theta        : TUNED ON A VALIDATION SET to maximise F1-5

METRICS (QAMPARI's own)
  Precision, Recall, F1, and Recall-5 = min(hits,5)/min(|gold|,5), with
  F1-5 the harmonic mean of Precision and Recall-5.

THE ONE UNAVOIDABLE CHOICE
  The paper composes with "the same LLM L" that produced the samples. With a
  heterogeneous pool there is no single L. We use the strongest pool member as
  the composer and record it in the output. Any choice here is a deviation
  forced by the substitution, not a modification of the method.

Usage:
  ./venv/bin/python asc_paper.py                 # full run (composes via API)
  ./venv/bin/python asc_paper.py --no-compose    # steps 1-5 only, no API calls
"""
import sys, os, json, collections, statistics, random, argparse

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
from open_ended_aggregation.benchmarks import qampari as Q

GEN = f"{HERE}/data/qampari_asc800.jsonl"
COMPOSED = f"{HERE}/data/qampari_asc800_composed.jsonl"
COMPOSER = "grok-4.3"
SEED = 0

# The paper's P_combine, quoted from the method section.
P_COMBINE = ("Remove irrelevant sentences and combine all relevant ones into a "
             "single answer that can address all interpretations of the question. "
             "Do not miss any minor details relevant to the question. "
             "Output only the list of answers, one per line.")


# ---------------------------------------------------------------- edit distance
def lev(a, b, cutoff):
    """Levenshtein with early exit once every cell in a row exceeds cutoff."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(a) - len(b) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            if v < best:
                best = v
        if best > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


def ned(a, b, tau_max=1.0):
    """Normalised edit distance in [0,1]. tau_max bounds the work: any pair that
    cannot come in under tau_max returns 1.0 without full computation."""
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    if abs(len(a) - len(b)) / m >= tau_max:      # length alone rules it out
        return 1.0
    d = lev(a, b, int(tau_max * m))
    return d / m


# ---------------------------------------------------------------- data
def load(pool=None):
    recs = [json.loads(l) for l in open(GEN)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({r["model"] for r in recs}) if pool is None else sorted(pool)
    qids = sorted(q for q, d in byq.items() if set(models) <= set(d))
    items = {it["qid"]: it for it in Q.load_items(100000) if it["qid"] in set(qids)}
    return byq, models, [q for q in qids if q in items], items


# ---------------------------------------------------------------- steps 1-3
def edges(byq, q, models, tau_max):
    """Pairwise structure for one question, computed ONCE and reused at every tau.

    Identical normalised strings are collapsed first: they are always at distance
    0, so they necessarily share a cluster. Clustering the UNIQUE keys and
    carrying multiplicity into the member count is exact and removes most of the
    O(n^2) work -- agents repeat the same entities heavily.
    """
    units = []
    for m in models:
        for it in byq[q][m]["items"]:
            k = Q.norm(it)
            if k:
                units.append((k, it))
    order, mult, first = [], collections.Counter(), {}
    for k, it in units:
        if k not in first:
            first[k] = it; order.append(k)
        mult[k] += 1
    E = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            d = ned(order[i], order[j], tau_max)
            if d < tau_max:
                E.append((d, i, j))
    return order, mult, first, E


def cluster_at(order, mult, first, E, tau):
    """Union-find over the cached edge list at threshold tau.
    Representative = FIRST member in encounter order. Strength = member count."""
    parent = list(range(len(order)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for d, i, j in E:
        if d < tau:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    mem = collections.defaultdict(list)
    for i in range(len(order)):
        mem[find(i)].append(i)
    return [(sum(mult[order[i]] for i in v), first[order[min(v)]]) for v in mem.values()]


# ---------------------------------------------------------------- metrics
def score(pred, gold_sets):
    P = [Q.norm(p) for p in pred]
    hit_gold = sum(1 for gs in gold_sets if any(p in gs for p in P))
    hit_pred = sum(1 for p in P if any(p in gs for gs in gold_sets))
    prec = hit_pred / max(1, len(P))
    rec = hit_gold / max(1, len(gold_sets))
    rec5 = min(hit_gold, 5) / max(1, min(len(gold_sets), 5))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    f15 = 0.0 if prec + rec5 == 0 else 2 * prec * rec5 / (prec + rec5)
    return dict(prec=prec, rec=rec, f1=f1, rec5=rec5, f15=f15)


def goldsets(item):
    return [{Q.norm(a) for a in ([g["answer_text"]] + list(g.get("aliases") or []))
             if str(a).strip()} for g in item["gold"]]


# ---------------------------------------------------------------- composition
def compose_all(need, byq, items):
    """Step 6. Cached and resumable -- one API call per question."""
    import azure_backend as AZ, hashlib
    from concurrent.futures import ThreadPoolExecutor

    def sig(reps):
        """Composition depends on WHICH representatives survived the filter, so
        the cache key must include them. Keying on qid alone would silently reuse
        an answer composed under a different Theta or a different clustering --
        the generations are config-independent, this step is not."""
        return hashlib.sha1(" || ".join(reps).encode()).hexdigest()[:16]

    cached = {}
    if os.path.exists(COMPOSED):
        for l in open(COMPOSED):
            try:
                r = json.loads(l)
                cached[(r["qid"], r.get("sig"))] = r["items"]
            except Exception:
                pass
    done = {q: cached[(q, sig(reps))] for q, reps in need if (q, sig(reps)) in cached}
    todo = [(q, reps) for q, reps in need if q not in done]
    if not todo:
        print(f"  composition cached for all {len(need)} questions", flush=True)
        return done
    print(f"  composing {len(todo)} questions with {COMPOSER} "
          f"({len(done)} cached)", flush=True)

    def one(a):
        q, reps = a
        user = (f"Question: {items[q]['question']}\n\n"
                f"Candidate answers:\n" + "\n".join(reps))
        txt, _ = AZ.chat(COMPOSER, P_COMBINE, user, 4000, temp=0)
        return q, (Q.parse_list(txt) if txt else None)

    f = open(COMPOSED, "a")
    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for q, out in ex.map(one, todo):
            n += 1
            if out is not None:
                done[q] = out
                f.write(json.dumps({"qid": q, "composer": COMPOSER,
                                    "sig": sig(dict(todo)[q]), "items": out}) + "\n")
                f.flush()
            if n % 100 == 0:
                print(f"    composed {n}/{len(todo)}  ${AZ.spend():.3f}", flush=True)
    f.close()
    return done


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compose", action="store_true")
    ap.add_argument("--tau", type=float, default=None,
                    help="fix the edit-distance threshold instead of tuning it")
    a = ap.parse_args()

    byq, models, qids, items = load()
    print("=" * 74)
    print(f"  ASC AS PUBLISHED, agents substituted for samples")
    print(f"  n={len(qids)} questions   agents={len(models)}   m={len(models)}")
    print("=" * 74)
    for m in models:
        print(f"    {m}")

    gs = {q: goldsets(items[q]) for q in qids}
    rng = random.Random(SEED)
    shuf = qids[:]; rng.shuffle(shuf)
    val, test = shuf[:len(shuf) // 2], shuf[len(shuf) // 2:]
    print(f"\n  validation {len(val)}   test {len(test)}   (Theta and tau tuned on validation)")

    taus = [a.tau] if a.tau else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    import time
    t0 = time.time()
    E = {}
    for n_, q in enumerate(qids, 1):
        E[q] = edges(byq, q, models, max(taus))
        if n_ % 100 == 0:
            print(f"    clustered {n_}/{len(qids)}  {time.time()-t0:.0f}s", flush=True)
    cache = {}
    for tau in taus:
        cache[tau] = {q: cluster_at(*E[q], tau) for q in qids}
        print(f"  tau={tau:.2f}: mean clusters/question "
              f"{statistics.mean(len(cache[tau][q]) for q in qids):.1f}", flush=True)

    def arm(tau, theta, qs):
        return [score([rep for c, rep in cache[tau][q] if c >= theta], gs[q]) for q in qs]

    best = None
    thetas = list(range(1, len(models) * 3 + 1))
    for tau in taus:
        for th in thetas:
            v = statistics.mean(s["f15"] for s in arm(tau, th, val))
            if best is None or v > best[0]:
                best = (v, tau, th)
    _, TAU, TH = best
    print(f"\n  tuned on validation:  tau={TAU}  Theta={TH}  (max F1-5)")

    print(f"\n  {'arm':<34}{'P':>8}{'R':>8}{'F1':>8}{'R-5':>8}{'F1-5':>8}")
    print("  " + "-" * 76)

    def show(lbl, rows):
        print(f"  {lbl:<34}" + "".join(
            f"{100*statistics.mean(r[k] for r in rows):8.2f}"
            for k in ("prec", "rec", "f1", "rec5", "f15")))

    for m in models:
        show(f"single agent: {m[:20]}",
             [score(byq[q][m]["items"], gs[q]) for q in test])
    show("union (no filter)", arm(TAU, 1, test))
    show(f"ASC filtered (Theta={TH}), no compose", arm(TAU, TH, test))

    if not a.no_compose:
        need = [(q, [rep for c, rep in cache[TAU][q] if c >= TH]) for q in qids]
        comp = compose_all(need, byq, items)
        got = [q for q in test if q in comp]
        show(f"ASC AS PUBLISHED (composed)", [score(comp[q], gs[q]) for q in got])
        print(f"\n  composed coverage on test: {len(got)}/{len(test)}")
        json.dump({"n_test": len(got), "tau": TAU, "theta": TH,
                   "composer": COMPOSER, "agents": models,
                   "asc_composed": {k: 100*statistics.mean(score(comp[q], gs[q])[k] for q in got)
                                    for k in ("prec","rec","f1","rec5","f15")}},
                  open(f"{HERE}/results/asc_paper.json", "w"), indent=2)
        print("  wrote results/asc_paper.json")


if __name__ == "__main__":
    main()
