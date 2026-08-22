"""
Is QUEST's gold too incomplete to trust?

THE WORRY
  QUEST gold is derived from Wikipedia category intersections, so an entity that
  genuinely satisfies the query but was never filed under the source category is
  scored WRONG. Measured symptom: P(correct) rises with agreement up to 7 of 8
  agents (27.7%) and then FALLS at unanimity (17.6%) -- the opposite of QAMPARI,
  where it rises monotonically to 56.5%. Items every model agrees on are famous
  entities, and famous entities are what category-derived gold misses.

  If that is right, "MA-count does nothing on QUEST" may be a statement about the
  labels rather than about consensus.

THE TEST
  Ask a judge OUTSIDE the agent pool whether an item actually satisfies the
  query, on three samples:

    HIGH-CONSENSUS, NOT IN GOLD   >=6 agents agree, scored wrong  <- the suspects
    IN GOLD                       scored right                    <- judge calibration
    SOLO, NOT IN GOLD             1 agent only, scored wrong      <- false-positive rate

  Without the two controls the result is uninterpretable: a judge that says yes
  to everything would "prove" the gold is broken.

  This is a DIAGNOSTIC of label quality, not a re-grading. Headline metrics stay
  programmatic -- introducing a judge there would forfeit the judge-free property
  that makes these benchmarks usable.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.quest_gold_audit [n_per_group]
"""
import sys, json, random, collections
from concurrent.futures import ThreadPoolExecutor

from open_ended_aggregation.benchmarks import quest as QU
from open_ended_aggregation.backends import azure as AZ
from open_ended_aggregation.paths import DATA, RESULTS

JUDGE = "gpt-5.5"          # deliberately NOT one of the eight agents
SEED = 0

PROMPT = (
    "You verify whether a candidate entity satisfies a query.\n"
    "Answer YES only if the candidate genuinely satisfies EVERY condition in the "
    "query, including negations and conjunctions. Answer NO if it violates any "
    "condition. Answer UNSURE if you do not know the entity or cannot tell.\n"
    "Judge the real-world fact, not whether it appears on any particular list.\n"
    "Reply with exactly one word: YES, NO, or UNSURE."
)


def main(per_group=50):
    items = {it["qid"]: it for it in QU.load_items(400)}
    # QUEST rows carry no qid; load_items assigns quest:test:%05d in file order
    # after the same 5..40 answer filter, so replay that to recover original_query.
    raw = [json.loads(l) for l in open(DATA / "quest_test.jsonl") if l.strip()]
    ordered = [r for r in raw if 5 <= len(r.get("docs") or []) <= 40]
    orig = {f"quest:test:{i:05d}": r for i, r in enumerate(ordered)}

    recs = [json.loads(l) for l in open(QU.GEN_PATH)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    POOL = sorted({r["model"] for r in recs})
    qids = [q for q, d in byq.items() if set(POOL) <= set(d) and q in items]

    hi, gold_ctl, solo = [], [], []
    for q in qids:
        gset = {QU.norm(g) for g in items[q]["gold"]}
        who = collections.defaultdict(set)
        rep = {}
        for m in POOL:
            for it in byq[q][m]["items"]:
                k = QU.norm(it)
                if k:
                    who[k].add(m); rep.setdefault(k, it)
        for k, ms in who.items():
            entry = (q, rep[k], len(ms))
            if k in gset:
                gold_ctl.append(entry)
            elif len(ms) >= 6:
                hi.append(entry)
            elif len(ms) == 1:
                solo.append(entry)

    rng = random.Random(SEED)
    groups = {
        "HIGH-CONSENSUS, not in gold": rng.sample(hi, min(per_group, len(hi))),
        "IN GOLD (calibration)":       rng.sample(gold_ctl, min(per_group, len(gold_ctl))),
        "SOLO, not in gold":           rng.sample(solo, min(per_group, len(solo))),
    }
    print(f"  judge = {JUDGE} (outside the agent pool)")
    print(f"  pool sizes: high-consensus non-gold {len(hi)}, "
          f"in-gold {len(gold_ctl)}, solo non-gold {len(solo)}")

    def ask(a):
        q, item, n = a
        o = orig.get(q, {})
        user = (f"Query: {items[q]['question']}\n"
                f"Precise form: {o.get('original_query','')}\n\n"
                f"Candidate: {item}\n\nDoes it satisfy the query?")
        txt, _ = AZ.chat(JUDGE, PROMPT, user, 2000, temp=None)
        v = (txt or "").strip().upper()
        return "YES" if v.startswith("YES") else "NO" if v.startswith("NO") else "UNSURE"

    out = {}
    for name, sample in groups.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            verdicts = list(ex.map(ask, sample))
        c = collections.Counter(verdicts)
        n = len(verdicts)
        out[name] = {k: c[k] for k in ("YES", "NO", "UNSURE")}
        print(f"\n  {name}  (n={n})")
        for k in ("YES", "NO", "UNSURE"):
            print(f"     {k:<7}{c[k]:>4}  {100*c[k]/max(1,n):5.1f}%")
        if name.startswith("HIGH"):
            for (q, it, k), v in list(zip(sample, verdicts))[:6]:
                print(f"       [{v:<6}] {it[:44]:46s} {items[q]['question'][:44]}")

    json.dump(out, open(RESULTS / "quest_gold_audit.json", "w"), indent=2)
    print(f"\n  wrote results/quest_gold_audit.json")
    print(f"  spend ${AZ.spend():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
