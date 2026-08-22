"""
What ASQA actually measures -- and the correction that follows.

WHAT WE HAD BEEN DOING, AND WHY IT WAS WRONG
  Earlier analysis scored ASQA with STR-EM alone and concluded the benchmark
  "cannot adjudicate a merging filter because its metric is recall-only". STR-EM
  is recall-only -- but it is one COMPONENT, not ASQA's metric.

  ASQA (Stelmakh et al., EMNLP 2022, arXiv:2204.06092) reports:
    ROUGE-L       lexical overlap with the human long_answer, max over the two
                  references
    STR-EM        fraction of disambiguations whose short answer appears
    Disambig-F1   a RoBERTa-SQuADv2 reader answers each disambiguated question
                  using the prediction as context; token-F1 against gold
    DR            GEOMETRIC MEAN of ROUGE-L and Disambig-F1  <- the headline

  The authors state the geometric mean is chosen "to penalize methods that
  maximize one metric at the cost of a significant decrease in the other", and
  that STR-EM and Disambig-F1 measure the same aspect so only one enters DR.

  ROUGE-L is therefore the precision/length term we claimed ASQA did not have.
  Concatenating every agent's response maximises STR-EM and destroys ROUGE-L, so
  under DR the union is NOT free -- which is the whole question.

WHAT THIS FILE COMPUTES
  ROUGE-L (LCS F-measure, max over references), STR-EM, and DR-proxy =
  sqrt(ROUGE-L * STR-EM). The proxy substitutes STR-EM for Disambig-F1 on the
  authors' own statement that the two measure the same aspect; it avoids standing
  up a RoBERTa reader. It is a PROXY and is labelled as such -- a real
  Disambig-F1 needs the reader model.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.asqa_metrics [gen_file]
"""
import sys, json, math, collections, statistics

import pandas as pd

from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.paths import DATA


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, refs):
    """LCS F-measure, max over references. This is the term that punishes
    answers far longer than the reference: recall stays high but precision
    collapses as the prediction grows."""
    p = A.norm(pred).split()
    best = 0.0
    for r in refs:
        g = A.norm(r).split()
        if not p or not g:
            continue
        l = lcs(p, g)
        if l == 0:
            continue
        prec, rec = l / len(p), l / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def main(gen=None):
    gen = gen or str(DATA / "asqa_imb.jsonl")
    df = pd.read_parquet(DATA / "asqa_dev.parquet")
    refs = {}
    for i, row in df.iterrows():
        refs[f"asqa:{i:04d}"] = [a["long_answer"] for a in row["annotations"]
                                 if a.get("long_answer")]

    items = {it["qid"]: it for it in A.load_items(100000)}
    A._ITEMS.update(items)
    recs = [json.loads(l) for l in open(gen)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    POOL = sorted({r["model"] for r in recs})
    qids = [q for q, d in byq.items() if len(d) == len(POOL) and q in items and refs.get(q)]

    print("=" * 78)
    print(f"  ASQA METRICS   {len(qids)} questions, {len(POOL)} agents")
    print(f"  source: {gen}")
    print("=" * 78)

    def report(label, textof):
        se = [A.str_em(textof(q), items[q]["short_sets"]) for q in qids]
        rl = [rouge_l(textof(q), refs[q]) for q in qids]
        dr = [math.sqrt(max(0.0, a) * max(0.0, b)) for a, b in zip(rl, se)]
        wl = statistics.mean(len(textof(q).split()) for q in qids)
        print(f"  {label:<34}{100*statistics.mean(se):8.2f}{100*statistics.mean(rl):9.2f}"
              f"{100*statistics.mean(dr):8.2f}{wl:9.0f}")
        return statistics.mean(dr)

    print(f"\n  {'arm':<34}{'STR-EM':>8}{'ROUGE-L':>9}{'DR*':>8}{'words':>9}")
    print("  " + "-" * 68)

    best_m, best_v = None, -1
    for m in POOL:
        v = report(f"single: {m[:26]}", lambda q, m=m: byq[q][m]["resp"])
        if v > best_v:
            best_m, best_v = m, v
    print()
    report("UNION (concatenate all agents)",
           lambda q: " ".join(byq[q][m]["resp"] for m in POOL))

    # sentence-level count filter, the ASC-style arm
    import re
    SENT = re.compile(r"(?<=[.!?])\s+")

    def sents(t):
        return [s.strip() for s in SENT.split(str(t)) if len(s.strip().split()) >= 3]

    def filtered(q, theta):
        seen = collections.Counter()
        rep = {}
        for m in POOL:
            for s in {A.norm(x): x for x in sents(byq[q][m]["resp"])}.items():
                seen[s[0]] += 1; rep.setdefault(s[0], s[1])
        keep = [rep[k] for k, c in seen.items() if c >= theta]
        return " ".join(keep)

    print()
    for th in range(1, len(POOL) + 1):
        report(f"exact-sentence count filter th={th}", lambda q, t=th: filtered(q, t))

    print(f"\n  DR* = sqrt(ROUGE-L x STR-EM), a PROXY for the published DR score, which")
    print(f"        uses Disambig-F1 (a RoBERTa-SQuADv2 reader) in place of STR-EM.")
    print(f"  best single agent by DR*: {best_m} ({100*best_v:.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
