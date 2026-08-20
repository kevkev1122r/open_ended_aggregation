"""
Unattended driver: generate -> judge -> analyse -> report.

Written to run while nobody is watching, which changes what it has to do:

  * EVERY STAGE IS RESUMABLE. Generation and judging both key on (qid, model) and
    skip what already exists, so a crash or a tripped cap costs only the calls
    that had not happened yet. Re-running this script continues rather than
    restarts.
  * IT NEVER SILENTLY PROCEEDS ON PARTIAL DATA. If generation stops early the
    judge and analysis still run, but the report says exactly how complete the
    matrix was. A missing cell must be visible in the writeup, not inferred.
  * THE OPENROUTER CAP IS THE REAL GUARD. Azure is grant credit; OpenRouter is
    ~$8 of actual money covering Anthropic, OpenAI and the judge. Generation gets
    its own cap below the balance so the judge still has room to run afterwards.

Usage:  ./venv/bin/python run_all.py [per_domain] [or_cap]
"""
import os, sys, json, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PER      = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OR_CAP   = float(sys.argv[2]) if len(sys.argv) > 2 else 6.50
TOTAL_CAP = 40.0
GEN      = "data/v2.jsonl"
JUDGED   = "data/v2_judged.jsonl"


def log(msg):
    print(f"\n{'='*78}\n  {msg}\n{'='*78}", flush=True)


def main():
    import benchmarks as B
    import run_v2, judge as J
    import openrouter_backend as OR, azure_backend as AZ

    items = B.load_all(PER, overrides={"mmlupro": PER * 2})
    qmap = {i["qid"]: i["question"] for i in items}
    or_start = OR.spend()

    log(f"STAGE 1 / GENERATE   {len(items)} questions x {len(run_v2.MODELS)} models"
        f"   OpenRouter cap +${OR_CAP:.2f}")
    try:
        run_v2.run(PER, TOTAL_CAP, OR_CAP)
    except Exception:
        traceback.print_exc()
        print("  generation raised -- continuing with whatever was written", flush=True)

    gen_path = os.path.join(HERE, GEN)
    if not os.path.exists(gen_path):
        print("  no generation output at all -- stopping", flush=True)
        return
    rows = [json.loads(l) for l in open(gen_path)]
    print(f"\n  {len(rows)} rows generated"
          f"   OpenRouter spent +${OR.spend()-or_start:.3f}", flush=True)

    # read the model, never restate it -- a hardcoded label that drifts from the
    # code is how a log ends up asserting something false about its own run
    log(f"STAGE 2 / JUDGE   {J.JUDGE} (lab has no model in the pool)")
    try:
        J.run(gen_path, os.path.join(HERE, JUDGED), qmap)
    except Exception:
        traceback.print_exc()
        print("  judging raised -- analysis will fall back to string grading", flush=True)

    log("STAGE 3 / ANALYSE")
    try:
        import analyze_domains as A
        d = A.load(gen=GEN, judged=JUDGED)
        piv, dom = A.complete_matrix(d)
        print(f"  {len(d):,} responses   graded by: {d.graded_by.value_counts().to_dict()}")
        print(f"  complete (every model answered): {len(piv):,} of "
              f"{d.qid.nunique():,} questions", flush=True)
        acc = A.capability(piv, dom)
        M, p = A.specialisation(piv, dom)
        res = A.aggregation(d, piv, dom)
        os.makedirs(A.OUT, exist_ok=True)
        json.dump({"per_domain": PER, "n_rows": len(d),
                   "n_complete": int(len(piv)), "n_questions": int(d.qid.nunique()),
                   "overall": acc.to_dict(), "matrix": M.to_dict(),
                   "interaction_p": p, "aggregation": res,
                   "openrouter_spent": OR.spend() - or_start},
                  open(os.path.join(A.OUT, "domain_analysis.json"), "w"),
                  indent=2, default=float)
        print("\n  -> results/domain_analysis.json", flush=True)
    except Exception:
        traceback.print_exc()

    log(f"DONE   OpenRouter total this run +${OR.spend()-or_start:.3f}"
        f"   Azure (reconstructed) ${AZ.spend():.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  wall clock {(time.time()-t0)/60:.1f} min", flush=True)
