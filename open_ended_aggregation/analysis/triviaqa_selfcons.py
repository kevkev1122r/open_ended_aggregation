"""
TriviaQA: does per-agent CONFIDENCE beat per-agent RELIABILITY on a single-answer task?

THE GAP THIS FILLS
  rank+budget cannot run on TriviaQA -- each agent emits one answer, so rank is
  always 0, the keep-count is always 1, and no agent is ever silent. All three
  components are degenerate. But the underlying CLAIM is not about lists, it is:

      a signal about how confidently an agent said something beats a global
      estimate of how reliable that agent is.

  TriviaQA can test that claim, because the cache holds TWO responses per agent
  per question -- a `terse` one (1.9 words) and a `natural` one (23 words).

  SELF-CONSISTENCY.  Does the agent's terse answer appear inside its own natural
  sentence? Label-free, needs no LLM canonicalisation, and it is the genuine ASC
  idea applied within an agent rather than across agents.

  VERBOSITY.  How long the natural response is. Hedging ("I cannot verify who
  played...") is longer than a confident answer, so length should carry signal
  the same way list length does on QAMPARI.

ARMS   (selection, not filtering -- pick one answer, score = accuracy)
  best single | majority vote | OW log-odds weighted vote  [paper-exact]
  | learned base | +self-consistency | +verbosity | +both | oracle

  OW log-odds is applicable here and only here: TriviaQA per-agent accuracy is
  0.57-0.86, so log(p/(1-p)) is positive. On QAMPARI per-claim precision is
  5.8-28%, every weight goes negative, and the method inverts.

  5-fold cross-fitted over questions; weights and coefficients fitted on train.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.triviaqa_selfcons
"""
import json, re, math, random, argparse, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
FOLDS = 5
_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def load():
    recs = [json.loads(l) for l in open(DATA / "gen_full.jsonl")]
    by = collections.defaultdict(dict)
    for r in recs:
        by[(r["qid"], r["model"])].setdefault(r["style"], r)
    models = sorted({r["model"] for r in recs})
    qids = sorted({q for q, _ in by})
    keep = [q for q in qids
            if all(len(by.get((q, m), {})) >= 2 for m in models)]
    return models, keep, by


def main():
    models, qids, by = load()
    n = len(models)
    mi = {m: i for i, m in enumerate(models)}
    print("=" * 88)
    print("  TRIVIAQA -- self-consistency and verbosity vs OW reliability weighting")
    print("=" * 88)
    print(f"\n  {len(qids)} questions with both styles for all {n} agents\n")

    # ---- per (question, agent): terse answer, correctness, self-consistency, length
    ans, ok, sc, wl = {}, {}, {}, {}
    for q in qids:
        for m in models:
            t = by[(q, m)]["terse"]; nat = by[(q, m)]["natural"]
            a = norm(t["resp"])
            ans[(q, m)] = a
            ok[(q, m)] = bool(t["correct"])
            nt = norm(nat["resp"])
            sc[(q, m)] = 1.0 if (a and a in nt) else 0.0
            wl[(q, m)] = math.log1p(len(str(nat["resp"]).split()))

    # ---- diagnostic: does self-consistency predict correctness?
    tab = collections.defaultdict(lambda: [0, 0])
    for q in qids:
        for m in models:
            tab[sc[(q, m)]][0] += ok[(q, m)]; tab[sc[(q, m)]][1] += 1
    print("  DIAGNOSTIC -- terse answer appears inside the agent's own natural answer?")
    for v in (1.0, 0.0):
        h, t = tab[v]
        lbl = "self-consistent" if v else "NOT consistent"
        print(f"    {lbl:<18}{100*t/sum(x[1] for x in tab.values()):6.1f}% of rows"
              f"   P(correct) {100*h/max(1,t):5.1f}%")
    r1 = tab[1.0][0] / max(1, tab[1.0][1]); r0 = tab[0.0][0] / max(1, tab[0.0][1])
    print(f"    ratio {r0/max(1e-9,r1):.2f}\n")

    qb = collections.defaultdict(lambda: [0, 0])
    for q in qids:
        for m in models:
            b = min(int(wl[(q, m)] / 0.7), 6)
            qb[b][0] += ok[(q, m)]; qb[b][1] += 1
    print("  DIAGNOSTIC -- natural-response length vs correctness")
    for b in sorted(qb):
        h, t = qb[b]
        if t > 200:
            print(f"    ~{int(math.exp(b*0.7)):>4} words   n={t:>6}   "
                  f"P(correct) {100*h/t:5.1f}%")
    print()

    # ---- candidates: distinct terse answers per question
    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}

    ARMS = ["majority", "OW log-odds", "learned base", "+self-cons",
            "+verbosity", "+both"]
    per = {a: {} for a in ARMS}

    for f in range(FOLDS):
        tr = [q for q in qids if fold[q] != f]
        te = [q for q in qids if fold[q] == f]
        acc = {m: (sum(ok[(q, m)] for q in tr) + 1) / (len(tr) + 2) for m in models}
        ow = {m: math.log(acc[m] / (1 - acc[m])) for m in models}

        def feats(q, sup):
            x = np.zeros(3 * n + 8)
            j = 0
            x[j] = 1.0; j += 1
            for m in sup:
                x[j + mi[m]] = 1.0
            j += n
            x[j] = len(sup) / n
            x[j + 1] = sum(ow[m] for m in sup) / 5.0
            j += 2
            s = [sc[(q, m)] for m in sup]
            x[j] = sum(s) / n
            x[j + 1] = statistics.mean(s)
            j += 2
            L = [wl[(q, m)] for m in sup]
            x[j] = statistics.mean(L) / 4.0
            x[j + 1] = min(L) / 4.0
            x[j + 2] = max(L) / 4.0
            j += 3
            for m in sup:
                x[j + mi[m]] = sc[(q, m)]
            j += n
            for m in sup:
                x[j + mi[m]] = wl[(q, m)] / 4.0
            return x

        BLOCKS = {"learned base": list(range(0, n + 3)),
                  "+self-cons": list(range(0, n + 3)) + list(range(n + 3, n + 5))
                                + list(range(n + 8, 2 * n + 8)),
                  "+verbosity": list(range(0, n + 3)) + list(range(n + 5, n + 8))
                                + list(range(2 * n + 8, 3 * n + 8)),
                  "+both": list(range(0, 3 * n + 8))}

        cands = {}
        for q in qids:
            g = collections.defaultdict(list)
            for m in models:
                g[ans[(q, m)]].append(m)
            cands[q] = [(a, sup, any(ok[(q, m)] for m in sup))
                        for a, sup in g.items() if a]

        Xtr, ytr = [], []
        for q in tr:
            for a, sup, good in cands[q]:
                Xtr.append(feats(q, sup)); ytr.append(float(good))
        Xtr = np.array(Xtr); ytr = np.array(ytr)

        W = {}
        for arm, cols in BLOCKS.items():
            W[arm] = (cols, fit_logistic(Xtr[:, cols], ytr, ridge=1.0))

        for q in te:
            C = cands[q]
            best = max(C, key=lambda c: (len(c[1]), max(acc[m] for m in c[1])))
            per["majority"][q] = float(best[2])
            bw = max(C, key=lambda c: sum(ow[m] for m in c[1]))
            per["OW log-odds"][q] = float(bw[2])
            F = np.array([feats(q, c[1]) for c in C])
            for arm, (cols, w) in W.items():
                s = F[:, cols] @ w
                per[arm][q] = float(C[int(np.argmax(s))][2])

    singles = {m: [float(ok[(q, m)]) for q in qids] for m in models}
    bm = max(models, key=lambda m: statistics.mean(singles[m]))
    best_v = singles[bm]
    oracle = [float(any(ok[(q, m)] for m in models)) for q in qids]
    maj = [per["majority"][q] for q in qids]
    owv = [per["OW log-odds"][q] for q in qids]

    print(f"  {'arm':<26}{'acc':>7}{'vs best single':>25}{'vs majority':>25}"
          f"{'vs OW':>25}")
    print("  " + "-" * 108)

    def cell(v, ref):
        d, lo, hi = bootstrap(v, ref)
        return (f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+6.1f}%)"
                f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}")

    rows = ([(f"best single({bm.split('/')[-1][:14]})", best_v),
             ("majority vote", maj), ("OW log-odds  [exact]", owv)] +
            [(a, [per[a][q] for q in qids]) for a in
             ("learned base", "+self-cons", "+verbosity", "+both")] +
            [("oracle", oracle)])
    res = {}
    for lbl, v in rows:
        res[lbl] = 100 * statistics.mean(v)
        c0 = f"{'(reference)':>25}" if v is best_v else cell(v, best_v)
        c1 = f"{'(reference)':>25}" if v is maj else cell(v, maj)
        c2 = f"{'(reference)':>25}" if v is owv else cell(v, owv)
        print(f"  {lbl:<26}{res[lbl]:7.2f}{c0:>25}{c1:>25}{c2:>25}")
    print("\n  * = bootstrap 95% CI over questions excludes 0")

    json.dump(res, open(RESULTS / "triviaqa_selfcons.json", "w"), indent=2)
    print("  wrote results/triviaqa_selfcons.json")


if __name__ == "__main__":
    main()
