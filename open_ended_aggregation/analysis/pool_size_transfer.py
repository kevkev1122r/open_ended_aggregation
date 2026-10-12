"""
Transfer to unseen agents at every pool-size split (paper Table "tab:transfer").

QAMPARI, eight agents split into two disjoint groups, questions split into two
halves. The pool-invariant scorer ("self" mode, pool_transfer.Fitted) and its
budget are fitted on one group's answers to one half and applied to the other
group's answers to the other half: no shared agent, no shared question, no
target gold read.

  4->4   all 35 splits, both directions (70)   -- same setting as cut_rules E2
  3->5   all 56 three-agent source groups
  5->3   the same 56 splits, reversed

Per split it records the transferred scorer, counting with its threshold fitted
on the source (zero-label), and two references fitted on the TARGET group's own
gold: counting, and the paper's 92-feature scorer + budget (rank_crossover).

Replaces the 22 Sept one-off whose json.dump crashed on numpy bools, so the
3->5 / 5->3 rows existed only as terminal output.

Cached generations only -- CPU, zero API spend.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.pool_size_transfer
"""
import json, random, itertools, statistics
import numpy as np
from open_ended_aggregation.analysis.pool_transfer import load, view, Fitted, AZURE, GPT, SEED, boot
from open_ended_aggregation.analysis.rank_crossover import run_ensemble, tensors
from open_ended_aggregation.paths import RESULTS


def main():
    models, qids, cand, ngold, nlist = load("qampari", AZURE + GPT)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    rng = random.Random(SEED)
    o = list(range(nq))
    rng.shuffle(o)
    h = nq // 2
    folds = [(o[:h], o[h:]), (o[h:], o[:h])]

    splits = {"4->4": [], "3->5": [], "5->3": []}
    for A in itertools.combinations(range(8), 4):
        B = [j for j in range(8) if j not in A]
        splits["4->4"].append((list(A), B))   # complement is also in the list: both directions
    for A in itertools.combinations(range(8), 3):
        B = [j for j in range(8) if j not in A]
        splits["3->5"].append((list(A), B))
        splits["5->3"].append((B, list(A)))

    res = {"models": models, "nq": nq}
    for key, pairs in splits.items():
        rows = []
        for src, tgt in pairs:
            vS, vT = view(T, src), view(T, tgt)
            tr_, c0 = np.zeros(nq), np.zeros(nq)
            for tr, te in folds:
                trm = np.zeros(nq, bool)
                trm[tr] = True
                out, _ = Fitted(vS, trm, "self").apply(vT)
                tr_[te] = out["bud-src"][te]
                c0[te] = out["cnt-thr"][te]
            ref = run_ensemble(tgt, *T, nq, folds)
            d, lo, hi = boot(tr_, c0)
            d2, lo2, hi2 = boot(tr_, ref["rank+budget"])
            rows.append(dict(src=[models[j] for j in src], tgt=[models[j] for j in tgt],
                             transfer=float(100 * tr_.mean()), count_src=float(100 * c0.mean()),
                             count_tgt_gold=float(100 * ref["count"].mean()),
                             full_tgt_gold=float(100 * ref["rank+budget"].mean()),
                             vs_count=[float(d), float(lo), float(hi)],
                             vs_full=[float(d2), float(lo2), float(hi2)],
                             beats_count_tgt_gold=bool(tr_.mean() > ref["count"].mean())))
        m = lambda x: statistics.mean(r[x] for r in rows)
        summ = dict(n=len(rows), transfer=m("transfer"), count_src=m("count_src"),
                    count_tgt_gold=m("count_tgt_gold"), full_tgt_gold=m("full_tgt_gold"),
                    gain=statistics.mean(r["vs_count"][0] for r in rows),
                    sig_wins=sum(r["vs_count"][1] > 0 for r in rows),
                    sig_losses=sum(r["vs_count"][2] < 0 for r in rows),
                    beats_count_tgt_gold=sum(r["beats_count_tgt_gold"] for r in rows),
                    vs_full=statistics.mean(r["vs_full"][0] for r in rows))
        res[key] = dict(summary=summ, splits=rows)
        print(f"{key}: n={summ['n']}  transfer {summ['transfer']:.2f} | count(src) {summ['count_src']:.2f}"
              f" | count(tgt gold) {summ['count_tgt_gold']:.2f} | full(tgt gold) {summ['full_tgt_gold']:.2f}")
        print(f"      vs count {summ['gain']:+.2f}  sig wins {summ['sig_wins']}/{summ['n']}"
              f"  sig losses {summ['sig_losses']}/{summ['n']}"
              f" | beats count(tgt gold) {summ['beats_count_tgt_gold']}/{summ['n']}"
              f" | vs full {summ['vs_full']:+.2f}", flush=True)

    with open(RESULTS / "pool_size_transfer.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
