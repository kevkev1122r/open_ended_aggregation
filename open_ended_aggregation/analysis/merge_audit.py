"""
Task 2a: how good were the 18 August LLM merges, measured against gold aliases?
Cached outputs only -- no API calls.

The August run (methods/llm_cluster.py) asked an LLM to group a question's
candidate strings that name the same entity. QAMPARI gold lists aliases per
answer, so every merged pair where BOTH strings match gold can be adjudicated:

  correct merge   both strings match the SAME gold answer
  false merge     they match DIFFERENT gold answers (deletes a correct answer
                  under the representative-only design of August)
  recall          of the pairs of distinct strings that match the same gold
                  answer (true aliases), the share the LLM merged

Pairs with a string outside gold are not adjudicable and are counted apart.

ORDER CHECK. The groups index candidates in the order the August code built
them; that code's loader (asc_paper.py) no longer exists, so the order is
rebuilt (agents sorted, items in list order) with the AUGUST normaliser
(norm_august: before strip_special was added -- using today's QA.norm shifts
the order and 20% of questions go out of range). VERIFIED: 0 questions out of
range, and merged pairs far more string-similar than random pairs. If the
check fails, no number is reported.

CAVEAT. Under the August normaliser, leaked chat tokens made "<|START_TEXT|>X"
and "X" distinct strings. Those pairs are counted apart as leaked-token
variants: they are trivial merges, not aliases. See merge_audit.log for the
split numbers.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.merge_audit
"""
import collections, difflib, json, random, statistics

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.paths import DATA

FILES = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5", "gpt55"]


def norm_august(s):
    """QA.norm as it was on 18 Aug: before strip_special() was added, so leaked
    chat tokens survived as words. The August groups index keys built with it."""
    s = str(s).lower().replace("’", "'")
    s = QA._PUNC.sub(" ", s)
    s = QA._ART.sub(" ", s)
    return " ".join(s.split())


def main():
    byq = collections.defaultdict(dict)
    for l in open(DATA / "qampari_asc800.jsonl"):
        r = json.loads(l)
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({m for d in byq.values() for m in d})
    items = {it["qid"]: it for it in QA.load_items(100000)}

    for tag in FILES:
        path = DATA / f"qampari_asc800_llmclusters.{tag}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        sim_m, sim_r, bad_idx = [], [], 0
        corr = false = unk = art_corr = 0
        alias_pairs = alias_merged = art_pairs = art_merged = 0
        rng = random.Random(0)
        for r in rows:
            q = r["qid"]
            if q not in items or q not in byq:
                continue
            order, first = [], {}
            for m in models:
                if m not in byq[q]:
                    continue
                for it in byq[q][m]["items"] or []:
                    k = norm_august(it)
                    if k and k not in first:
                        first[k] = it
                        order.append(k)
            n = len(order)
            if any(i >= n for g in r["groups"] for i in g):
                bad_idx += 1
                continue
            gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                     if str(a).strip()} for g in items[q]["gold"]]
            gi = {k: next((j for j, gs in enumerate(gold) if QA.norm(first[k]) in gs), None)
                  for k in order}
            merged = set()
            for g in r["groups"]:
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        x, y = order[g[a]], order[g[b]]
                        merged.add(frozenset((x, y)))
                        sim_m.append(difflib.SequenceMatcher(None, x, y).ratio())
                        if gi[x] is not None and gi[y] is not None:
                            if QA.norm(first[x]) == QA.norm(first[y]):
                                art_corr += 1          # leaked-token variant, not an alias
                            elif gi[x] == gi[y]:
                                corr += 1
                            else:
                                false += 1
                        else:
                            unk += 1
            for _ in range(20):
                if n >= 2:
                    x, y = rng.sample(order, 2)
                    sim_r.append(difflib.SequenceMatcher(None, x, y).ratio())
            by_g = collections.defaultdict(list)
            for k, j in gi.items():
                if j is not None:
                    by_g[j].append(k)
            for ks in by_g.values():
                for a in range(len(ks)):
                    for b in range(a + 1, len(ks)):
                        hit = frozenset((ks[a], ks[b])) in merged
                        if QA.norm(first[ks[a]]) == QA.norm(first[ks[b]]):
                            art_pairs += 1; art_merged += hit
                        else:
                            alias_pairs += 1; alias_merged += hit
        print(f"\n  clusterer {tag}: {len(rows)} questions ({bad_idx} skipped: index out of range)")
        if not sim_m:
            print("    no merged pairs")
            continue
        ok_order = statistics.mean(sim_m) > statistics.mean(sim_r) + 0.2
        print(f"    ORDER CHECK: string similarity of merged pairs {statistics.mean(sim_m):.2f} vs "
              f"random pairs {statistics.mean(sim_r):.2f} -> "
              f"{'reconstruction OK' if ok_order else 'RECONSTRUCTION WRONG -- ignore below'}")
        if not ok_order:
            continue
        dec = corr + false
        print(f"    merged pairs: {len(sim_m)}; adjudicable REAL pairs {dec} -> correct {corr}, "
              f"FALSE {false}  (merge precision {100*corr/max(1,dec):.1f}%); "
              f"leaked-token variants merged {art_corr}; not adjudicable {unk}")
        print(f"    real alias pairs: {alias_pairs}, merged {alias_merged} "
              f"(recall {100*alias_merged/max(1,alias_pairs):.1f}%);  leaked-token variant pairs: "
              f"{art_pairs}, merged {art_merged}")


if __name__ == "__main__":
    main()
