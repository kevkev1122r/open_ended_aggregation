"""
Every ensemble of size 3..6 drawn from the 8 agents, on QAMPARI.

THE QUESTION (from the 19 Aug meeting)
  For each ensemble, do we improve on (a) the best single model in that ensemble,
  and (b) self-consistency/counting -- and how does that change with ensemble
  size and composition? The full-pool numbers may hide the answer.

  C(8,3)+C(8,4)+C(8,5)+C(8,6) = 56+70+56+28 = 210 ensembles.

HONEST THRESHOLDS
  Thresholds are cross-fitted (2-fold): chosen on one half, scored on the other,
  both directions pooled. This matters -- on QAMPARI, tuning the threshold on the
  evaluation set makes weighting look like +0.08 against counting, while
  cross-fitting shows -1.25. Anything tuned in-sample is not a result.

IMPLEMENTATION
  A candidate's support is a BITMASK over the 8 agents, so restricting to an
  ensemble is a bitwise AND. Within an ensemble, candidates are grouped by their
  restricted pattern -- there are at most 2^k of those -- so scoring a filter
  costs O(2^k) per question instead of O(#candidates). A threshold rule can only
  realise <= 2^k distinct filters, so all of them are enumerated exactly rather
  than swept.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.ensemble_sizes
"""
import json, math, itertools, collections, statistics, random

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0


def load():
    recs = [json.loads(l) for l in open(DATA / "qampari_asc800.jsonl")]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    POOL = sorted({r["model"] for r in recs})
    items = {it["qid"]: it for it in QA.load_items(100000)}
    qids = sorted(q for q, d in byq.items() if set(POOL) <= set(d) and q in items)
    bit = {m: 1 << i for i, m in enumerate(POOL)}

    cand, ngold = {}, {}
    for q in qids:
        gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                 if str(a).strip()} for g in items[q]["gold"]]
        ngold[q] = len(gold)
        who = collections.defaultdict(int)
        for m in POOL:
            for it in byq[q][m]["items"]:
                k = QA.norm(it)
                if k:
                    who[k] |= bit[m]
        rows = []
        for k, mask in who.items():
            gi = next((i for i, gs in enumerate(gold) if k in gs), None)
            rows.append((mask, gi))
        cand[q] = rows
    return POOL, bit, qids, cand, ngold


def group(rows, emask):
    """pattern -> (n_items, n_correct, frozenset(gold indices hit))"""
    g = collections.defaultdict(lambda: [0, 0, set()])
    for mask, gi in rows:
        r = mask & emask
        if not r:
            continue
        e = g[r]
        e[0] += 1
        if gi is not None:
            e[1] += 1; e[2].add(gi)
    return g


def f1_from(groups, keep, ngold):
    n = c = 0
    hit = set()
    for pat in keep:
        e = groups.get(pat)
        if e:
            n += e[0]; c += e[1]; hit |= e[2]
    if n == 0:
        return 0.0
    p = c / n
    r = len(hit) / max(1, ngold)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    POOL, bit, qids, cand, ngold = load()
    print(f"  QAMPARI  {len(qids)} questions, pool of {len(POOL)}")
    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]

    singles = {}
    for m in POOL:
        b = bit[m]
        singles[m] = {q: f1_from(group(cand[q], b), [b], ngold[q]) for q in qids}

    out = []
    for k in (3, 4, 5, 6):
        for ens in itertools.combinations(POOL, k):
            emask = 0
            for m in ens:
                emask |= bit[m]
            groups = {q: group(cand[q], emask) for q in qids}
            pats = [p for p in range(1, 1 << len(POOL)) if p & ~emask == 0]

            # per-agent precision within this ensemble, estimated on a train half
            def weights(train):
                n = collections.Counter(); c = collections.Counter()
                for q in train:
                    for pat, e in groups[q].items():
                        for m in ens:
                            if pat & bit[m]:
                                n[m] += e[0]; c[m] += e[1]
                return {m: (c[m] + 1) / (n[m] + 2) for m in ens}

            scored = {"count": {}, "weighted": {}}
            for train, test in folds:
                w = weights(train)
                rules = {
                    "count": lambda p, _w=w: bin(p).count("1"),
                    "weighted": lambda p, _w=w: sum(_w[m] for m in ens if p & bit[m]),
                }
                for arm, fn in rules.items():
                    ths = sorted({fn(p) for p in pats})
                    best = None
                    for t in ths:
                        keep = [p for p in pats if fn(p) >= t - 1e-12]
                        v = statistics.mean(f1_from(groups[q], keep, ngold[q]) for q in train)
                        if best is None or v > best[0]:
                            best = (v, t)
                    keep = [p for p in pats if fn(p) >= best[1] - 1e-12]
                    for q in test:
                        scored[arm][q] = f1_from(groups[q], keep, ngold[q])

            bm = max(ens, key=lambda m: statistics.mean(singles[m][q] for q in qids))
            best_v = [singles[bm][q] for q in qids]
            cnt = [scored["count"][q] for q in qids]
            wgt = [scored["weighted"][q] for q in qids]
            d1, l1, h1 = bootstrap(cnt, best_v)
            d2, l2, h2 = bootstrap(wgt, best_v)
            d3, l3, h3 = bootstrap(wgt, cnt)
            prec = {m: statistics.mean(singles[m][q] for q in qids) for m in ens}
            v = sorted(prec.values(), reverse=True)
            out.append(dict(k=k, ens=list(ens), best_model=bm,
                            best=100*statistics.mean(best_v),
                            count=100*statistics.mean(cnt),
                            weighted=100*statistics.mean(wgt),
                            cnt_vs_best=d1, cnt_sig=(l1 > 0 or h1 < 0),
                            wgt_vs_best=d2, wgt_sig=(l2 > 0 or h2 < 0),
                            wgt_vs_cnt=d3, wgt_cnt_sig=(l3 > 0 or h3 < 0),
                            dominance=v[0] / max(1e-9, statistics.mean(v[1:]))))
        print(f"    size {k}: {sum(1 for r in out if r['k']==k)} ensembles done", flush=True)

    print(f"\n  {'size':>5}{'n':>5}{'best single':>13}{'count':>9}{'weighted':>10}"
          f"{'cnt-best':>11}{'wgt-best':>11}{'wgt-cnt':>10}")
    print("  " + "-" * 76)
    for k in (3, 4, 5, 6):
        R = [r for r in out if r["k"] == k]
        print(f"  {k:>5}{len(R):>5}{statistics.mean(r['best'] for r in R):>13.2f}"
              f"{statistics.mean(r['count'] for r in R):>9.2f}"
              f"{statistics.mean(r['weighted'] for r in R):>10.2f}"
              f"{statistics.mean(r['cnt_vs_best'] for r in R):>+11.2f}"
              f"{statistics.mean(r['wgt_vs_best'] for r in R):>+11.2f}"
              f"{statistics.mean(r['wgt_vs_cnt'] for r in R):>+10.2f}")

    print(f"\n  how often each arm SIGNIFICANTLY beats the alternative")
    print(f"  {'size':>5}{'count > best':>15}{'weighted > best':>17}{'weighted > count':>18}")
    for k in (3, 4, 5, 6):
        R = [r for r in out if r["k"] == k]
        a = sum(1 for r in R if r["cnt_sig"] and r["cnt_vs_best"] > 0)
        b = sum(1 for r in R if r["wgt_sig"] and r["wgt_vs_best"] > 0)
        c = sum(1 for r in R if r["wgt_cnt_sig"] and r["wgt_vs_cnt"] > 0)
        print(f"  {k:>5}{a:>10}/{len(R):<4}{b:>12}/{len(R):<4}{c:>13}/{len(R):<4}")

    xs = [r["dominance"] for r in out]; ys = [r["wgt_vs_cnt"] for r in out]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    print(f"\n  corr(relative dominance, weighted-count) over all {len(out)} ensembles"
          f" = {num/den:+.3f}")

    json.dump(out, open(RESULTS / "ensemble_sizes.json", "w"), indent=1)
    print("  wrote results/ensemble_sizes.json")


if __name__ == "__main__":
    main()
