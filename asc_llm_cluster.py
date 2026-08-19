"""
LLM-judged clustering as a drop-in replacement for ASC's surface-form step.

THIS IS A VARIANT, NOT ASC. Published ASC clusters list items by normalised edit
distance (arXiv:2405.13131). Everything else in the pipeline is kept identical --
atomic units, member-count strength, Theta tuned on validation, LLM composition --
so the only moving part is how equivalence is decided. asc_paper.py keeps the
unmodified baseline; this file is the ablation against it.

COST
  One call per question, NOT O(n^2) pairwise. The model receives the whole
  candidate list and returns groups. ~800 calls for the full run instead of the
  ~5.5M pairwise judgements a naive implementation would need.

CLUSTERER CHOICE
  Set with ASC_CLUSTERER (default gpt-5.4-mini). A SMALL model is used on
  purpose: deciding whether two strings name the same entity is far easier than
  generating the list, so capability spent here is mostly wasted.

  Caveat: the small models available are also pool agents, so the clusterer
  adjudicates equivalence among its own outputs. That is milder than it sounds --
  clustering only decides GROUPING; correctness is still scored programmatically
  against gold aliases, so a model cannot promote its own wrong answers. But it
  can merge its own surface variants preferentially, and that should be stated.

  Note this decides CLUSTERING only. Grading stays programmatic against gold
  aliases -- introducing an LLM there would forfeit QAMPARI's judge-free
  property, which is the reason this benchmark is the cleanest one available.

Usage:  ./venv/bin/python asc_llm_cluster.py [--limit N]
"""
import sys, os, json, re, collections, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_qampari as Q

CACHE = f"{HERE}/data/qampari_asc800_llmclusters.{os.environ.get('ASC_CLUSTERER','gpt-5.4-mini')}.jsonl"
CLUSTERER = os.environ.get("ASC_CLUSTERER", "gpt-5.4-mini")

PROMPT = (
    "You group answer candidates that refer to THE SAME real-world entity.\n"
    "Two candidates belong together if they name the same entity in different "
    "surface forms: abbreviations, alternate spellings, transliterations, "
    "with or without a type suffix (e.g. 'Essendon' and 'Essendon Football Club'), "
    "with or without a disambiguator, or differing only in punctuation.\n"
    "Do NOT group two genuinely different entities, even if their names are "
    "similar (e.g. 'Melbourne Football Club' and 'Melbourne Victory' are "
    "DIFFERENT; 'Richmond' and 'Richmond Football Club' are the SAME).\n"
    "Most candidates belong to NO group -- they are distinct entities. Group only "
    "when you are confident two names denote the identical entity.\n"
    "A group is normally 2-3 members and never more than 6. Never put many "
    "different entities in one group.\n"
    "Output one group per line as comma-separated numbers. Include only groups "
    "with two or more members. Output nothing else."
)


MAX_GROUP = 6      # surface variants of ONE entity; anything larger is a failure

def parse_groups(txt, n):
    """Guarded parse.

    Small models fail this task in a specific way: they emit one line listing
    every index, which naive parsing turns into a single giant cluster. Measured
    2026-08-18, gpt-5.4-nano collapsed 130+ distinct Scottish wind farms into one
    group, which deletes an entire answer list. A legitimate group is a handful
    of surface forms of ONE entity, so any line proposing more than MAX_GROUP
    members is discarded rather than trusted.
    """
    groups = []
    seen = set()
    dropped = 0
    for line in str(txt).splitlines():
        idx = [int(x) - 1 for x in re.findall(r"\d+", line)]
        idx = [i for i in idx if 0 <= i < n and i not in seen]
        if len(idx) > MAX_GROUP:
            dropped += 1
            continue
        if len(idx) >= 2:
            groups.append(idx)
            seen.update(idx)
    return groups


def load_cache():
    out = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            try:
                r = json.loads(l); out[r["qid"]] = r["groups"]
            except Exception:
                pass
    return out


def build(byq, qids, models, items, limit=None):
    """Returns {qid: [[key_index,...], ...]} over the question's UNIQUE norm keys."""
    from concurrent.futures import ThreadPoolExecutor
    import azure_backend as AZ

    keys = {}
    for q in qids:
        order, first = [], {}
        for m in models:
            for it in byq[q][m]["items"]:
                k = Q.norm(it)
                if k and k not in first:
                    first[k] = it; order.append(k)
        keys[q] = (order, first)

    done = load_cache()
    todo = [q for q in qids if q not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"  LLM clusters cached for all {len(qids)} questions", flush=True)
        return done, keys
    print(f"  clustering {len(todo)} questions with {CLUSTERER} "
          f"({len(done)} cached)", flush=True)

    def one(q):
        order, first = keys[q]
        listing = "\n".join(f"{i+1}. {first[k]}" for i, k in enumerate(order))
        user = f"Question: {items[q]['question']}\n\nCandidates:\n{listing}"
        txt, _ = AZ.chat(CLUSTERER, PROMPT, user, 4000, temp=None)
        return q, (parse_groups(txt, len(order)) if txt else None)

    f = open(CACHE, "a")
    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for q, g in ex.map(one, todo):
            n += 1
            if g is not None:
                done[q] = g
                f.write(json.dumps({"qid": q, "clusterer": CLUSTERER, "groups": g}) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"    {n}/{len(todo)}  ${AZ.spend():.3f}", flush=True)
    f.close()
    return done, keys


def clusters_from(byq, q, models, order, first, groups):
    """Same output shape as asc_paper.cluster_at: [(member_count, representative)].
    Multiplicity is preserved -- strength is the COUNT OF MEMBERS, as the paper
    defines it, not the number of distinct agents."""
    mult = collections.Counter()
    for m in models:
        for it in byq[q][m]["items"]:
            k = Q.norm(it)
            if k:
                mult[k] += 1
    parent = list(range(len(order)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for g in groups:
        for i in g[1:]:
            ri, rj = find(g[0]), find(i)
            if ri != rj:
                parent[ri] = rj
    mem = collections.defaultdict(list)
    for i in range(len(order)):
        mem[find(i)].append(i)
    return [(sum(mult[order[i]] for i in v), first[order[min(v)]]) for v in mem.values()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import asc_paper as A
    byq, models, qids, items = A.load()
    print(f"  {len(qids)} questions, {len(models)} agents")
    groups, keys = build(byq, qids, models, items, a.limit)
    have = [q for q in qids if q in groups]
    merged = sum(sum(len(g) - 1 for g in groups[q]) for q in have)
    tot = sum(len(keys[q][0]) for q in have)
    print(f"\n  clustered {len(have)} questions")
    print(f"  keys merged away: {merged}/{tot} ({100*merged/max(1,tot):.1f}%)")
    ex = [q for q in have if groups[q]][:3]
    for q in ex:
        order, first = keys[q]
        print(f"\n  {items[q]['question'][:70]}")
        for g in groups[q][:4]:
            print("     ", [first[order[i]] for i in g])
