"""
INDEPENDENT re-derivation of the QAMPARI merging result.

Written per HANDOFF_CURRENT.md §9: re-derive the +1.90 from data/qampari_gen.jsonl
WITHOUT reading analyze_merge.py. Nothing here is copied from that file. The
method spec is taken from the handoff prose; the grading contract (norm, score_set)
is taken from run_qampari.py, which is the *generation*-side code that produced the
already-trusted per-model baselines.

Run:  ./venv/bin/python verify_qampari_independent.py
"""
import json, re, zipfile, random, collections, statistics, os

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
random.seed(0)

# ---------------------------------------------------------------- grading contract
# Reproduced from run_qampari.py so that merged sets are graded on exactly the
# same terms as the per-model baselines already in the jsonl.
_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def score_set(pred_items, gold_sets):
    P = [norm(p) for p in pred_items]
    hit_gold = sum(1 for gs in gold_sets if any(p in gs for p in P))
    hit_pred = sum(1 for p in P if any(p in gs for gs in gold_sets))
    rec = hit_gold / max(1, len(gold_sets))
    prec = hit_pred / max(1, len(P))
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- data
def load():
    recs = [json.loads(l) for l in open(f"{HERE}/data/qampari_gen.jsonl")]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]][r["model"]] = r
    models = sorted({r["model"] for r in recs})
    qids = sorted(q for q, d in byq.items() if len(d) == len(models))

    z = zipfile.ZipFile(f"{HERE}/data/qampari.zip")
    with z.open("qampari_data/dev_data.jsonl") as f:
        gold = {}
        want = set(qids)
        for l in f:
            r = json.loads(l)
            if r["qid"] in want:
                gold[r["qid"]] = [
                    {norm(a) for a in ([g["answer_text"]] + list(g.get("aliases") or []))
                     if str(a).strip()}
                    for g in r["answer_list"]
                ]
    assert set(gold) == set(qids), f"gold missing for {len(set(qids)-set(gold))} qids"
    return byq, models, qids, gold


# ---------------------------------------------------------------- merging
def support(byq, q, models, weights):
    """Weight mass behind each normalised candidate item.

    Support is counted ONCE PER MODEL. The premise being tested is 'how many
    distinct models assert this item', so a model that lists an item twice
    (possible when two distinct surface forms collapse under norm()) must not
    vote twice. This is the handoff's Bug A.
    """
    acc = collections.defaultdict(float)
    surface = {}
    for m in models:
        seen = set()
        for it in byq[q][m]["items"]:
            k = norm(it)
            if not k or k in seen:
                continue
            seen.add(k)
            acc[k] += weights[m]
            surface.setdefault(k, it)
    return acc, surface


def merged_f1(byq, qids, models, gold, weight_fn, theta):
    out = []
    for q in qids:
        acc, surface = support(byq, q, models, weight_fn(q))
        keep = [surface[k] for k, v in acc.items() if v >= theta - 1e-12]
        out.append(score_set(keep, gold[q]))
    return out


# ---------------------------------------------------------------- weights
def crossfit_precision(byq, qids, models, folds=5):
    """Per-model precision estimated out-of-fold, so a question is never scored
    with a weight that saw its own label."""
    idx = list(range(len(qids)))
    random.Random(0).shuffle(idx)
    fold_of = {qids[i]: j % folds for j, i in enumerate(idx)}
    w = {}
    for f in range(folds):
        train = [q for q in qids if fold_of[q] != f]
        w[f] = {m: statistics.mean(byq[q][m]["prec"] for q in train) for m in models}
    return fold_of, w


def bootstrap(a, b, n=10000, seed=0):
    """Paired percentile CI on mean(a)-mean(b), resampling questions."""
    rng = random.Random(seed)
    k = len(a)
    d = [x - y for x, y in zip(a, b)]
    idx = range(k)
    reps = []
    for _ in range(n):
        s = [d[rng.choice(idx)] for _ in range(k)]
        reps.append(sum(s) / k)
    reps.sort()
    return (statistics.mean(d) * 100,
            reps[int(0.025 * n)] * 100,
            reps[int(0.975 * n)] * 100)


def show(name, diff, lo, hi):
    star = " *" if (lo > 0 or hi < 0) else ""
    print(f"  {name:34s} {diff:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]{star}")


# ---------------------------------------------------------------- main
def main():
    byq, models, qids, gold = load()
    n = len(qids)
    print("=" * 74)
    print(f"  INDEPENDENT QAMPARI RE-DERIVATION   n={n}  models={len(models)}")
    print("=" * 74)

    # --- sanity: does our re-grading of each model's own list reproduce the
    # generation-time f1 field? If not, the grading contract has drifted.
    print("\n[0] grading-contract check (re-grade each model's raw list)")
    worst = 0.0
    for m in models:
        mine = statistics.mean(score_set(byq[q][m]["items"], gold[q]) for q in qids)
        theirs = statistics.mean(byq[q][m]["f1"] for q in qids)
        worst = max(worst, abs(mine - theirs))
        print(f"  {m:32s} regraded {mine*100:6.2f}   stored {theirs*100:6.2f}   "
              f"delta {abs(mine-theirs)*100:.4f}")
    print(f"  max |delta| = {worst*100:.4f}  ->  "
          f"{'OK' if worst < 1e-9 else 'MISMATCH — everything below is suspect'}")

    singles = {m: [score_set(byq[q][m]["items"], gold[q]) for q in qids] for m in models}
    best_m = max(models, key=lambda m: statistics.mean(singles[m]))
    best = singles[best_m]
    print(f"\n  best single model = {best_m}   F1 {statistics.mean(best)*100:.2f}")

    # --- how bad is Bug A here? count within-model duplicate collapses
    dup = sum(len(byq[q][m]["items"]) - len({norm(i) for i in byq[q][m]["items"]})
              for q in qids for m in models)
    tot = sum(len(byq[q][m]["items"]) for q in qids for m in models)
    print(f"  within-model duplicates under norm(): {dup} / {tot} items "
          f"({dup/tot*100:.2f}%)  <- Bug A's raw material")

    # --- weights
    fold_of, wf = crossfit_precision(byq, qids, models)
    glob = {m: statistics.mean(byq[q][m]["prec"] for q in qids) for m in models}
    print("\n  cross-fitted precision weights (fold 0 shown; global in parens):")
    for m in sorted(models, key=lambda m: -glob[m]):
        print(f"    {m:32s} {wf[0][m]:.4f}   ({glob[m]:.4f})")
    solo = sorted(glob.values(), reverse=True)
    pairs = sorted(glob[a] + glob[b] for i, a in enumerate(models) for b in models[i+1:])
    print(f"    strongest solo {solo[0]:.4f}   2nd solo {solo[1]:.4f}   "
          f"weakest pair {pairs[0]:.4f}")
    print("    -> weighting can only differ from counting for theta in "
          f"({solo[1]:.4f}, {pairs[0]:.4f}]")

    W_CF = lambda q: wf[fold_of[q]]
    W_ONE = lambda q: {m: 1.0 for m in models}

    # --- arms
    print("\n[1] COUNT filter (integer support)")
    count_arms = {}
    for t in range(1, len(models) + 1):
        s = merged_f1(byq, qids, models, gold, W_ONE, t)
        count_arms[t] = s
        print(f"  theta={t}   F1 {statistics.mean(s)*100:6.2f}")
    union = count_arms[1]
    best_count_t = max(count_arms, key=lambda t: statistics.mean(count_arms[t]))
    count = count_arms[best_count_t]

    print("\n[2] WEIGHTED filter, fine sweep (step 0.005)")
    grid = [round(0.05 + 0.005 * i, 4) for i in range(int((1.20 - 0.05) / 0.005) + 1)]
    sweep = []
    for t in grid:
        s = merged_f1(byq, qids, models, gold, W_CF, t)
        sweep.append((statistics.mean(s), t, s))
    sweep.sort(key=lambda r: -r[0])
    top = sweep[0]
    weighted, theta_star = top[2], top[1]
    print(f"  best theta = {theta_star}   F1 {top[0]*100:6.2f}")
    print("  neighbourhood:")
    for mu, t, _ in sorted([r for r in sweep if 0.20 <= r[1] <= 0.40], key=lambda r: r[1]):
        mark = "  <<<" if t == theta_star else ""
        print(f"    theta={t:.3f}  F1 {mu*100:6.2f}{mark}")

    print("\n[3] CONTROLS")
    uni_sweep = [(statistics.mean(merged_f1(byq, qids, models, gold, W_ONE, t)), t)
                 for t in [round(0.25 + 0.05 * i, 3) for i in range(40)]]
    uni_best = max(uni_sweep)
    uniform = merged_f1(byq, qids, models, gold, W_ONE, uni_best[1])
    print(f"  UNIFORM weights   best theta {uni_best[1]}  F1 {uni_best[0]*100:6.2f}")

    vals = [glob[m] for m in models]
    shuf_runs = []
    for seed in range(20):
        rng = random.Random(100 + seed)
        perm = vals[:]
        while True:
            rng.shuffle(perm)
            if all(perm[i] != vals[i] for i in range(len(vals))):
                break   # derangement: every model gets someone else's weight
        wmap = dict(zip(models, perm))
        best_s = max(
            (statistics.mean(merged_f1(byq, qids, models, gold, lambda q: wmap, t)), t)
            for t in [round(0.10 + 0.02 * i, 3) for i in range(50)]
        )
        shuf_runs.append(best_s[0])
    print(f"  SHUFFLED weights  mean over 20 derangements  "
          f"F1 {statistics.mean(shuf_runs)*100:6.2f}   "
          f"(min {min(shuf_runs)*100:.2f}, max {max(shuf_runs)*100:.2f})")

    # one fixed derangement, kept paired for a CI
    rng = random.Random(100)
    perm = vals[:]
    while True:
        rng.shuffle(perm)
        if all(perm[i] != vals[i] for i in range(len(vals))):
            break
    wmap = dict(zip(models, perm))
    sh_best = max(
        (statistics.mean(merged_f1(byq, qids, models, gold, lambda q: wmap, t)), t)
        for t in [round(0.10 + 0.02 * i, 3) for i in range(50)]
    )
    shuffled = merged_f1(byq, qids, models, gold, lambda q: wmap, sh_best[1])

    print("\n[4] SUMMARY")
    print(f"  best single ({best_m})            F1 {statistics.mean(best)*100:6.2f}")
    print(f"  union, no filter                    F1 {statistics.mean(union)*100:6.2f}")
    print(f"  count filter (theta={best_count_t})              F1 {statistics.mean(count)*100:6.2f}")
    print(f"  WEIGHTED     (theta={theta_star})          F1 {statistics.mean(weighted)*100:6.2f}")
    print(f"  uniform control                     F1 {statistics.mean(uniform)*100:6.2f}")
    print(f"  shuffled control                    F1 {statistics.mean(shuffled)*100:6.2f}")

    print("\n[5] PAIRED BOOTSTRAP (10k, percentile, * = excludes 0)")
    show("count - best single", *bootstrap(count, best))
    show("WEIGHTED - best single", *bootstrap(weighted, best))
    show("WEIGHTED - count", *bootstrap(weighted, count))
    show("WEIGHTED - uniform", *bootstrap(weighted, uniform))
    show("WEIGHTED - shuffled", *bootstrap(weighted, shuffled))
    show("uniform - count", *bootstrap(uniform, count))

    print("\n[6] WHAT THE WEIGHTED RULE ACTUALLY IS at theta*")
    w0 = wf[0]
    passes_solo = [m for m in models if w0[m] >= theta_star]
    all_pairs = all(w0[a] + w0[b] >= theta_star
                    for i, a in enumerate(models) for b in models[i+1:])
    print(f"  models whose SOLO assertion clears theta={theta_star}: "
          f"{passes_solo or 'none'}")
    print(f"  every pair clears theta: {all_pairs}")
    if all_pairs and passes_solo:
        print(f"  => the rule is exactly:  (>=2 models agree)  OR  "
              f"({' or '.join(passes_solo)} says it)")


if __name__ == "__main__":
    main()
