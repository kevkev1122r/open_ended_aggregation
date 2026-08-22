"""
Alternative weighting mechanisms, cross-fitted, on cached generations only.

WHY THE CURRENT WEIGHT IS THE WRONG QUANTITY
  The deployed weight is a model's MARGINAL precision, P(correct | j asserts).
  But the count filter already keeps everything >=2 models assert, so the only
  decision weighting changes is whether to admit a model's SOLO claims. The
  quantity that governs that is P(correct | j asserts AND nobody else does),
  which is very different: measured on QAMPARI the best agent is 28.7% overall
  but only 9.6% alone -- below the 20.4% that ANY two agreeing agents give.

  A second defect: non-assertion currently contributes exactly zero. If four
  strong models omit an item, that is evidence against it, and the present rule
  ignores it entirely.

SCHEMES COMPARED  (all thresholds and all parameters cross-fitted 5-fold;
                   nothing is estimated on the questions it is scored on)

  count            keep if >= theta models assert          [the ASC-style baseline]
  marginal         sum of P(correct | j asserts)           [what we deploy now]
  solo             sum of P(correct | j asserts alone)     [fixes the conditional]
  nb-assert        Naive Bayes log-LR, assertions only
  nb-full          Naive Bayes log-LR, assertions AND omissions
  pattern          shrunk lookup of P(correct | exact subset asserting)

  `pattern` is the most general rule expressible from the support pattern alone,
  so it upper-bounds count, marginal, solo and nb. If it beats them by little,
  model identity carries little information beyond the count and no reweighting
  will rescue the method. That is the diagnostic worth having.

Usage:
  ./venv/bin/python -m open_ended_aggregation.analysis.weighting_schemes [--pool N]
"""
import sys, json, math, argparse, collections, statistics, random, itertools

from open_ended_aggregation.benchmarks import qampari as QA, quest as QU
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
FOLDS = 5


# ------------------------------------------------------------------ data
def build(gen_path, loader, normf, goldf, pool=None):
    """Per question: [(key, frozenset(models asserting), gold_index or None)]."""
    recs = [json.loads(l) for l in open(gen_path)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({r["model"] for r in recs}) if pool is None else sorted(pool)
    items = {it["qid"]: it for it in loader}
    qids = sorted(q for q, d in byq.items() if set(models) <= set(d) and q in items)

    cand, ngold = {}, {}
    for q in qids:
        gold = goldf(items[q])
        ngold[q] = len(gold)
        who = collections.defaultdict(set)
        for m in models:
            for it in byq[q][m]["items"]:
                k = normf(it)
                if k:
                    who[k].add(m)
        rows = []
        for k, ms in who.items():
            gi = next((i for i, gs in enumerate(gold) if k in gs), None)
            rows.append((k, frozenset(ms), gi))
        cand[q] = rows
    return models, qids, cand, ngold


def f1_of(kept, ngold):
    """kept = list of gold_index-or-None for the items the filter kept."""
    if not kept:
        return 0.0
    hit_pred = sum(1 for g in kept if g is not None)
    hit_gold = len({g for g in kept if g is not None})
    prec = hit_pred / len(kept)
    rec = hit_gold / max(1, ngold)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ estimators
def est_marginal(rows, models):
    n = collections.Counter(); c = collections.Counter()
    for _, ms, gi in rows:
        for m in ms:
            n[m] += 1; c[m] += (gi is not None)
    return {m: (c[m] + 1) / (n[m] + 2) for m in models}


def est_solo(rows, models):
    n = collections.Counter(); c = collections.Counter()
    for _, ms, gi in rows:
        if len(ms) == 1:
            m = next(iter(ms)); n[m] += 1; c[m] += (gi is not None)
    return {m: (c[m] + 1) / (n[m] + 2) for m in models}


def est_nb(rows, models):
    """TPR_j = P(j asserts | correct), FPR_j = P(j asserts | wrong).
    Returns per-model (assert_weight, omit_weight) log-likelihood ratios plus
    the prior log-odds."""
    pos = sum(1 for _, _, gi in rows if gi is not None)
    neg = len(rows) - pos
    tp = collections.Counter(); fp = collections.Counter()
    for _, ms, gi in rows:
        for m in ms:
            (tp if gi is not None else fp)[m] += 1
    w = {}
    for m in models:
        tpr = (tp[m] + 1) / (pos + 2)
        fpr = (fp[m] + 1) / (neg + 2)
        w[m] = (math.log(tpr / fpr), math.log((1 - tpr) / (1 - fpr)))
    prior = math.log((pos + 1) / (neg + 1))
    return w, prior


def est_pattern(rows, models, alpha=20.0):
    """P(correct | exact subset) shrunk toward the estimate at that support COUNT.
    Patterns are sparse -- 2^n of them -- so an unshrunk lookup memorises noise."""
    by_pat = collections.defaultdict(lambda: [0, 0])
    by_cnt = collections.defaultdict(lambda: [0, 0])
    for _, ms, gi in rows:
        by_pat[ms][0] += (gi is not None); by_pat[ms][1] += 1
        by_cnt[len(ms)][0] += (gi is not None); by_cnt[len(ms)][1] += 1
    cnt_rate = {k: (v[0] + 1) / (v[1] + 2) for k, v in by_cnt.items()}
    out = {}
    for pat, (h, n) in by_pat.items():
        base = cnt_rate.get(len(pat), 0.05)
        out[pat] = (h + alpha * base) / (n + alpha)
    return out, cnt_rate


# ------------------------------------------------------------------ scorers
def make_scorers(train_rows, models):
    marg = est_marginal(train_rows, models)
    solo = est_solo(train_rows, models)
    nbw, prior = est_nb(train_rows, models)
    pat, cnt_rate = est_pattern(train_rows, models)

    def s_count(ms):    return float(len(ms))
    def s_marg(ms):     return sum(marg[m] for m in ms)
    def s_solo(ms):     return sum(solo[m] for m in ms)
    def s_nb_a(ms):     return prior + sum(nbw[m][0] for m in ms)
    def s_nb_f(ms):     return prior + sum(nbw[m][0] if m in ms else nbw[m][1]
                                           for m in models)
    def s_pat(ms):      return pat.get(ms, cnt_rate.get(len(ms), 0.05))
    return {"count": s_count, "marginal": s_marg, "solo": s_solo,
            "nb-assert": s_nb_a, "nb-full": s_nb_f, "pattern": s_pat}


def thresholds_for(scorer, rows, models):
    """Candidate thresholds = every distinct achievable score. With n models the
    support pattern takes 2^n values, so this is exact rather than a grid."""
    vals = {scorer(frozenset(c)) for r in range(len(models) + 1)
            for c in itertools.combinations(models, r)}
    return sorted(vals)


# ------------------------------------------------------------------ evaluation
def evaluate(name, gen, loader, normf, goldf, pool=None):
    models, qids, cand, ngold = build(gen, loader, normf, goldf, pool)
    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}

    per_arm = collections.defaultdict(dict)
    for f in range(FOLDS):
        tr = [q for q in qids if fold[q] != f]
        te = [q for q in qids if fold[q] == f]
        train_rows = [r for q in tr for r in cand[q]]
        scorers = make_scorers(train_rows, models)
        for arm, sc in scorers.items():
            ths = thresholds_for(sc, train_rows, models)
            best = None
            for t in ths:
                v = statistics.mean(
                    f1_of([gi for _, ms, gi in cand[q] if sc(ms) >= t - 1e-12], ngold[q])
                    for q in tr)
                if best is None or v > best[0]:
                    best = (v, t)
            th = best[1]
            for q in te:
                per_arm[arm][q] = f1_of(
                    [gi for _, ms, gi in cand[q] if sc(ms) >= th - 1e-12], ngold[q])

    # best single agent, same questions
    singles = {}
    for m in models:
        singles[m] = [f1_of([gi for _, ms, gi in cand[q] if m in ms], ngold[q])
                      for q in qids]
    bm = max(models, key=lambda m: statistics.mean(singles[m]))
    best = singles[bm]

    print(f"\n  {name}   n={len(qids)}  agents={len(models)}   "
          f"best single = {bm} ({100*statistics.mean(best):.2f})")
    print(f"    {'weighting scheme':<16}{'F1':>8}{'vs count':>22}{'vs best single':>22}")
    print("    " + "-" * 68)
    cnt = [per_arm["count"][q] for q in qids]
    res = {}
    for arm in ["count", "marginal", "solo", "nb-assert", "nb-full", "pattern"]:
        v = [per_arm[arm][q] for q in qids]
        res[arm] = 100 * statistics.mean(v)
        if arm == "count":
            cells = f"{'(reference)':>22}"
        else:
            d, l, h = bootstrap(v, cnt)
            cells = f"{d:+7.2f} [{l:+6.2f},{h:+6.2f}]{'*' if (l>0 or h<0) else ' '}"
        d2, l2, h2 = bootstrap(v, best)
        print(f"    {arm:<16}{100*statistics.mean(v):8.2f}{cells:>22}"
              f"{d2:+8.2f} [{l2:+6.2f},{h2:+6.2f}]{'*' if (l2>0 or h2<0) else ' '}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="", help="comma-separated models to exclude")
    a = ap.parse_args()
    drop = set(x for x in a.drop.split(",") if x)

    print("=" * 78)
    print("  ALTERNATIVE WEIGHTING SCHEMES -- 5-fold cross-fitted, cached data only")
    print("=" * 78)

    qa_gold = lambda it: [{QA.norm(x) for x in [g["answer_text"]] + list(g.get("aliases") or [])
                           if str(x).strip()} for g in it["gold"]]
    qu_gold = lambda it: [{QU.norm(g)} for g in it["gold"]]

    pool = None
    if drop:
        recs = [json.loads(l) for l in open(DATA / "qampari_asc800.jsonl")]
        pool = [m for m in sorted({r["model"] for r in recs}) if m not in drop]

    out = {}
    out["qampari"] = evaluate("QAMPARI", DATA / "qampari_asc800.jsonl",
                              QA.load_items(100000), QA.norm, qa_gold, pool)
    qpool = pool
    if drop:
        recs = [json.loads(l) for l in open(QU.GEN_PATH)]
        qpool = [m for m in sorted({r["model"] for r in recs}) if m not in drop]
    out["quest"] = evaluate("QUEST", QU.GEN_PATH, QU.load_items(400),
                            QU.norm, qu_gold, qpool)

    json.dump(out, open(RESULTS / "weighting_schemes.json", "w"), indent=2)
    print(f"\n  wrote results/weighting_schemes.json")


if __name__ == "__main__":
    main()
