"""
Long-form synthesis on ASQA: the full pipeline against plain synthesis and the
published baselines.

PIPELINE (all zero-label on the target pool)
  claims      results/claims_asqa.json from analysis/claim_extract.py (steps 1-2)
  scorer      methods/prose_features.ProseScorer fitted on the SOURCE agents'
              claims (gpt-5.4, -mini, -nano) on TRAIN-half questions; applied to
              the TARGET agents (DeepSeek-V4-Flash, MAI-Thinking-1, grok-4.3) on
              TEST-half questions. Pools share no agent, halves no question.
  composer    methods/composer.py, synthesizer openai/gpt-5.4 (not in the
              target pool, so it never merges its own answer)

ARMS  (all on the same test questions, same target answers, same order)
  plain           plain LLM synthesis (baseline everyone uses)
  table           + claim table with p, support and contradiction counts
  table_scramble  identical, p permuted across the question's claims (control)
  table_raw       the table without p (support/contradiction counts only)
  band            accepted / undecided / rejected facts from the scorer
  usc             Universal Self-Consistency: select one agent answer
  moa             Mixture-of-Agents aggregator, one layer (see methods/baselines.py)
  solo            the synthesizer alone, no answers (its own knowledge)
  (reference, no LLM)  each target agent alone; EXTRACTIVE = the target agents'
                  own sentences holding the highest-p facts, up to the same length
                  (the prose analogue of "scorer alone" on lists)

JUDGE (optional, --judge)  fact precision from methods/judge.py, same judge for
  every arm and the extractive reference; its own cache and --judge-cost cap.

METRICS  ASQA's: STR-EM (interpretation recall), ROUGE-L against the human long
  answers (the length/precision term), DR* = sqrt(ROUGE-L x STR-EM) (the proxy
  in analysis/asqa_metrics.py), and answer length. Paired bootstrap CIs.

SPEND  paid.PaidCaller: cached, capped, nothing sent without --confirm.

Usage:
  ./venv/bin/python -u -m open_ended_aggregation.analysis.longform_rungs            # estimate
  ./venv/bin/python -u -m open_ended_aggregation.analysis.longform_rungs --confirm  # spends
"""
import argparse, hashlib, json, math, statistics, sys

import numpy as np
import pandas as pd

from open_ended_aggregation.analysis.asqa_metrics import rouge_l
from open_ended_aggregation.analysis.claim_extract import load_asqa, sentences
from open_ended_aggregation.analysis.synth_rungs import qrng
from open_ended_aggregation.benchmarks import asqa as A
from open_ended_aggregation.methods import baselines as B
from open_ended_aggregation.methods import composer as C
from open_ended_aggregation.methods.claim_table import build
from open_ended_aggregation.methods import judge as J
from open_ended_aggregation.methods.prose_features import (ProseScorer, split_questions,
                                                            evaluate_transfer)
from open_ended_aggregation.paid import PaidCaller, NotConfirmed, CapReached
from open_ended_aggregation.paths import DATA, RESULTS

SRC = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
TGT = ["DeepSeek-V4-Flash", "MAI-Thinking-1", "grok-4.3"]
BLOCKS = {"A": (0, 50), "B": (50, 190)}   # positions in load_asqa's seeded order
ARMS = ["plain", "table", "table_scramble", "table_raw", "band", "usc", "moa", "solo"]
SOLO_ASK = "Answer the question. Write one paragraph of at most {w} words. Output only the answer.\n"


def stable_seed(q):
    """Same every run. Python's hash() is salted per process, which would change
    the scrambled prompt, miss the paid cache and pay again."""
    return int(hashlib.sha256(q.encode()).hexdigest()[:8], 16)


def references():
    df = pd.read_parquet(DATA / "asqa_dev.parquet")
    return {f"asqa:{i:04d}": [a["long_answer"] for a in row["annotations"] if a.get("long_answer")]
            for i, row in df.iterrows()}


def metrics(text, item, refs):
    se = A.str_em(text, item["short_sets"])
    rl = rouge_l(text, refs)
    return dict(str_em=se, rouge_l=rl, dr=math.sqrt(max(0, se) * max(0, rl)),
                words=len(str(text).split()))


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    reps = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.quantile(reps, .025), 100 * np.quantile(reps, .975)


def extractive_answer(q, byq, claims_by_q, pq, t, max_words):
    """No-LLM reference: sentences from the target agents that state the
    highest-p facts, in p order, until the length budget. Facts with p >= t
    first; if none clear t, the single best fact still gets its sentence."""
    claims, _ = build(claims_by_q[q], TGT)
    ranked = sorted(claims, key=lambda c: -pq.get(q, {}).get((c["slot"], c["value"]), 0.0))
    sent = {m: sentences(byq[q][m]["resp"]) for m in TGT}
    out, used, words = [], set(), 0
    for i, c in enumerate(ranked):
        p = pq.get(q, {}).get((c["slot"], c["value"]), 0.0)
        if p < t and i > 0:
            break
        m, k = min(((m, k) for m, k in c["supporters"].items() if k), key=lambda x: x[1],
                   default=(None, None))
        if m is None or not (1 <= k <= len(sent[m])) or (m, k) in used:
            continue
        s = sent[m][k - 1]
        if out and words + len(s.split()) > max_words:
            break
        out.append(s); used.add((m, k)); words += len(s.split())
    return " ".join(out)


def build_messages(q, arm, byq, items, claims_by_q, pq, t, max_words):
    """(system, user) for one arm, plus the target answers in the order shown."""
    agents = TGT[:]
    qrng(q, "agents").shuffle(agents)
    resp = [byq[q][m]["resp"] or "" for m in agents]
    question = items[q]["question"]
    if arm == "solo":
        return (None, SOLO_ASK.format(w=max_words) + f"\nQuestion: {question}"), agents
    if arm == "usc":
        return (None, B.usc_prompt(question, resp)), agents
    if arm == "moa":
        return B.moa_messages(question, resp), agents
    rec = claims_by_q[q]
    claims, _ = build(rec, TGT)
    return (None, C.prose_prompt(arm, question, resp, max_words, claims=claims,
                                 pq=pq.get(q, {}), names=C.slot_names(rec), t=t,
                                 seed=stable_seed(q) if arm == "table_scramble" else None)
            ), agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap on test questions (0 = all)")
    ap.add_argument("--synth", default="openai/gpt-5.4")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--judge", action="store_true", help="also judge fact precision")
    ap.add_argument("--judge-model", default=J.JUDGE_MODEL)
    ap.add_argument("--judge-cost", type=float, default=0.5)
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1],
                    help="1 swaps the halves: fit on the other half, test on this one")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--block", default="A", choices=list(BLOCKS),
                    help="question block; each block is its own 2-fold experiment, so "
                         "adding questions never changes (and re-pays) an earlier block")
    a = ap.parse_args()

    path = RESULTS / "claims_asqa.json"
    if not path.exists():
        sys.exit("  results/claims_asqa.json not found: run analysis/claim_extract first.")
    claims_by_q = json.load(open(path))
    byq, items, pool, order = load_asqa(None)
    lo, hi = BLOCKS[a.block]
    block = set(order[lo:hi])
    missing = block - set(claims_by_q)
    if missing:
        sys.exit(f"  block {a.block}: {len(missing)} questions have no claims yet -- "
                 f"run claim_extract --n {hi} first.")
    # the block alone: scorer, length norms and length instruction never see
    # questions outside it
    claims_by_q = {q: claims_by_q[q] for q in order[lo:hi]}
    assert set(SRC + TGT) <= set(pool), f"pool mismatch: {pool}"
    refs = references()

    # ---- zero-label scorer: source pool on train half, target pool on test half
    train_q, test_q = split_questions(claims_by_q)
    if a.fold:
        train_q, test_q = test_q, train_q
    sc = ProseScorer(claims_by_q, train_q, SRC, items)
    pq = sc.score(claims_by_q, test_q, TGT)
    tr = evaluate_transfer(claims_by_q, items, SRC, TGT, fold=a.fold)
    print(f"  scorer transfer (free diagnostic): {tr['n_claims']} target claims, base rate "
          f"{100*tr['base_rate']:.1f}%, AUC scorer {tr['auc_scorer']:.3f} vs support-only "
          f"{tr['auc_support']:.3f}; source threshold {sc.t:.3f}")

    test_q = [q for q in test_q if q in refs and refs[q]]
    if a.n:
        test_q = test_q[:a.n]
    max_words = int(np.median([len(r.split()) for q in test_q for r in refs[q]]))
    arms = a.arms.split(",")
    caller = PaidCaller(RESULTS / "longform_cache_asqa.jsonl", a.synth, a.effort,
                        cap=a.max_cost, confirm=a.confirm)
    msgs = {(q, arm): build_messages(q, arm, byq, items, claims_by_q, pq, sc.t, max_words)
            for q in test_q for arm in arms}
    print(f"  {len(test_q)} test questions x {len(arms)} arms; length instruction "
          f"{max_words} words (median ASQA reference)")
    caller.estimate([m for m, _ in msgs.values()], "long-form composer")
    jref = J.asqa_references() if a.judge else None
    judge = (PaidCaller(RESULTS / "longform_judge_cache_asqa.jsonl", a.judge_model, "low",
                        cap=a.judge_cost, confirm=a.confirm, json_mode=True) if a.judge else None)
    if judge:
        # Real judge prompts wherever the answer is already known (cached composer
        # outputs, the free extractive reference); a max_words placeholder only
        # for answers not yet composed -- so a rerun is not quoted as new spend.
        dummy = " ".join(["word"] * max_words)
        jp = []
        for q in test_q:
            for arm in arms:
                (sy, u), agents = msgs[(q, arm)]
                hit = caller.cache.get(caller._key(u, sy))
                txt = hit["text"] if hit else dummy
                if hit and arm == "usc":
                    j = B.usc_choice(txt, len(agents))
                    txt = byq[q][agents[j]]["resp"] if j is not None else dummy
                jp.append(J.long_judge_prompt(items[q]["question"], jref[q], txt))
            jp.append(J.long_judge_prompt(items[q]["question"], jref[q],
                                          extractive_answer(q, byq, claims_by_q, pq, sc.t,
                                                            max_words)))
        judge.estimate(jp, f"fact-precision judge ({a.judge_model})")
    if not a.confirm:
        print("  No --confirm: nothing was sent.")
        return

    side = RESULTS / f"longform_prompts_asqa{'_' + a.tag if a.tag else ''}{'' if a.block == 'A' else '_block' + a.block}_fold{a.fold}.jsonl"
    with open(side, "w") as fh:
        for (q, arm), ((sy, u), ag) in msgs.items():
            fh.write(json.dumps({"qid": q, "arm": arm, "system": sy, "user": u,
                                 "agents": ag}) + "\n")

    from concurrent.futures import ThreadPoolExecutor
    import collections

    def one(q):
        row, f = {}, collections.Counter()
        try:
            return _one(q, row, f)
        except (NotConfirmed, CapReached) as e:   # keep what this question already has
            f["cap:" + type(e).__name__] += 1
            return q, row, f

    def _one(q, row, f):
        for arm in arms:
            (sy, u), agents = msgs[(q, arm)]
            txt, meta = caller(u, tag=f"{q}|{arm}", system=sy)
            if not txt:
                f[arm] += 1
                continue
            if arm == "usc":
                j = B.usc_choice(txt, len(agents))
                if j is None:
                    f["usc_parse"] += 1
                    continue
                txt = byq[q][agents[j]]["resp"]
            row[arm] = dict(text=txt, **metrics(txt, items[q], refs[q]))
        for m in TGT:
            row["single:" + m] = metrics(byq[q][m]["resp"], items[q], refs[q])
        ext = extractive_answer(q, byq, claims_by_q, pq, sc.t, max_words)
        row["extractive"] = dict(text=ext, **metrics(ext, items[q], refs[q]))
        if judge:
            for arm in list(arms) + ["extractive"]:
                if arm in row and row[arm].get("text"):
                    jt, _ = judge(J.long_judge_prompt(items[q]["question"], jref[q],
                                                      row[arm]["text"]), tag=f"{q}|{arm}")
                    pr, ns, nc, nu = J.long_judge_scores(jt)
                    if pr is None and jt is None:
                        f["judge_" + arm] += 1
                    row[arm].update(judged_prec=pr, n_supported=ns, n_contra=nc, n_unver=nu)
        return q, row, f

    per_q, fails = {}, collections.Counter()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            for i, (q, row, f) in enumerate(ex.map(one, test_q), 1):
                fails.update(f)
                if all(arm in row for arm in arms):
                    per_q[q] = row
                if i % 5 == 0 or i == len(test_q):
                    print(f"    {i}/{len(test_q)}  paired {len(per_q)}  spend composer "
                          f"${caller.spent():.2f}  judge ${judge.spent() if judge else 0:.2f}  "
                          f"failures {dict(fails) or 0}", flush=True)
    except (NotConfirmed, CapReached) as e:
        print(f"  STOPPED: {e}")
    fails = dict(fails)

    report(per_q, arms, judged=bool(judge))
    json.dump({"arms": arms, "max_words": max_words, "scorer": tr, "per_question": per_q,
               "failures": fails, "spend": caller.spent(),
               "judge_spend": judge.spent() if judge else 0, "fold": a.fold, "block": a.block},
              open(RESULTS / f"longform_asqa{'_' + a.tag if a.tag else ''}{'' if a.block == 'A' else '_block' + a.block}_fold{a.fold}.json", "w"),
              indent=1)


def fpr(x):
    """Harmonic mean of judged fact precision and STR-EM (interpretation recall)."""
    p, r = x.get("judged_prec") or 0.0, x["str_em"]
    return 2 * p * r / (p + r) if p + r else 0.0


def report(per_q, arms, judged=False):
    if not per_q:
        print("  nothing paired -- no result")
        return
    qs = list(per_q)
    print(f"\n  ASQA long-form, {len(qs)} paired test questions (x100)")
    hdr = f"  {'arm':<26}{'DR*':>8}{'STR-EM':>9}{'ROUGE-L':>9}{'words':>8}"
    print(hdr + (f"{'judgedP':>9}{'F(P,R)':>8}" if judged else ""))
    cols = arms + ["extractive"] + [k for k in per_q[qs[0]] if k.startswith("single:")]
    # judged columns only on questions where EVERY judged arm has a judgement, so
    # arms are compared on the same questions (a judge cap or parse failure drops
    # the question for all arms, not for one)
    jcols = arms + ["extractive"]
    jq = [q for q in qs if all(per_q[q].get(c, {}).get("judged_prec") is not None
                               for c in jcols)] if judged else []
    if judged:
        print(f"  judged columns on {len(jq)}/{len(qs)} questions judged for every arm")
    for arm in cols:
        v = [per_q[q][arm] for q in qs if arm in per_q[q]]
        line = (f"  {arm:<26}{100*statistics.mean(x['dr'] for x in v):8.2f}"
                f"{100*statistics.mean(x['str_em'] for x in v):9.2f}"
                f"{100*statistics.mean(x['rouge_l'] for x in v):9.2f}"
                f"{statistics.mean(x['words'] for x in v):8.0f}")
        if judged and arm in jcols and jq:
            jp = [per_q[q][arm]["judged_prec"] for q in jq]
            fp = [fpr(per_q[q][arm]) for q in jq]
            line += f"{100*statistics.mean(jp):9.2f}{100*statistics.mean(fp):8.2f}"
        elif judged:
            line += f"{'-':>9}{'-':>8}"
        print(line)
    for x, y in (("table", "table_scramble"), ("table", "table_raw"), ("table", "plain"),
                 ("band", "plain"), ("plain", "extractive"), ("table", "extractive"),
                 ("plain", "usc"), ("plain", "moa"), ("table", "moa")):
        if x in arms and y in arms:
            d, lo, hi = boot([per_q[q][x]["dr"] for q in qs], [per_q[q][y]["dr"] for q in qs])
            print(f"  DR* {x} - {y:<16} {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]"
                  f"  {'SIG' if lo > 0 or hi < 0 else 'ns'}")
            if jq:
                d, lo, hi = boot([fpr(per_q[q][x]) for q in jq], [fpr(per_q[q][y]) for q in jq])
                print(f"  F(P,R) {x} - {y:<13} {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]"
                      f"  {'SIG' if lo > 0 or hi < 0 else 'ns'}")


if __name__ == "__main__":
    main()
