"""
Pool-invariant, label-free features for PROSE claims, and the zero-label scorer.

Each list feature (pool_transfer.inv_features) has a prose counterpart, plus
contradiction, which lists never had:

  support        |S|/k, and |S| / (max support of any value in the same slot)
  CONTRADICTION  |C|/k agents asserting a different value in the slot;
                 support margin (|S| - best competing value's support) / k;
                 number of distinct values in the slot
  omission       |silent|/k, and the silent agents' answer length (self-normalised)
  position       sentence index / answer length, min and mean over supporters;
                 log of the earliest absolute sentence index
  verbosity      supporters' answer length, self-normalised against each agent's
                 own median length over all questions (label-free)
  question       log #claims, log #slots, mean support on the question

ZERO-LABEL SCORER (the definition chosen 22 Sept: no labels for the new pool)
  fit on SOURCE agents' claims on TRAIN questions (labels = ASQA short-answer
  match), apply to TARGET agents' claims on TEST questions. Also returns the
  source-fitted threshold, which the band composer uses.
"""
import math, random

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.methods.claim_table import build, gold_aliases, is_correct

FEATURES = ["const", "support", "support_rel_slot", "contradict", "margin",
            "slot_values", "silent", "silent_len", "pos_min", "pos_mean", "first_sent",
            "sup_len", "q_claims", "q_slots", "q_mean_support"]


def median_len(claims_by_q, agents):
    """Each agent's median answer length in sentences, over all questions (no labels)."""
    lens = {m: [] for m in agents}
    for rec in claims_by_q.values():
        for m in agents:
            if rec["n_sent"].get(m):
                lens[m].append(rec["n_sent"][m])
    return {m: max(1.0, float(np.median(v))) if v else 1.0 for m, v in lens.items()}


def featurize(claims_by_q, qids, agents, items=None, merge_contra=False):
    """Rows for every claim of `agents` on `qids`. Returns X, y (None without
    items), rowq (question index), meta (qid, slot, value).

    merge_contra=True is the ablation for "contradiction != omission": agents
    that gave a different value are treated as SILENT, and the contradiction
    columns carry no information (as in a list-style claim table)."""
    k = len(agents)
    med = median_len(claims_by_q, agents)
    X, y, rowq, meta = [], [], [], []
    for qi, q in enumerate(qids):
        rec = claims_by_q.get(q)
        if not rec:
            continue
        claims, slot_values = build(rec, agents)
        if not claims:
            continue
        al = gold_aliases(items[q]) if items else None
        mean_sup = np.mean([len(c["supporters"]) for c in claims]) / k
        for c in claims:
            S, C, Sil = c["supporters"], c["contradict"], c["silent"]
            vals = slot_values[c["slot"]]
            best_other = max([n for v, n in vals.items() if v != c["value"]], default=0)
            if merge_contra:
                Sil, C = sorted(set(Sil) | set(C)), []
                vals, best_other = {c["value"]: len(S)}, 0
            rel = [(S[m] / max(1, c["n_sent"][m])) for m in S if S[m]]
            first = min([S[m] for m in S if S[m]], default=1)
            X.append([
                1.0,
                len(S) / k,
                len(S) / max(vals.values()),
                len(C) / k,
                (len(S) - best_other) / k,
                math.log(len(vals)),
                len(Sil) / k,
                np.mean([math.log(max(1, c["n_sent"][m]) / med[m]) for m in Sil]) if Sil else 0.0,
                min(rel) if rel else 1.0,
                float(np.mean(rel)) if rel else 1.0,
                math.log1p(first) / 3.0,
                float(np.mean([math.log(max(1, c["n_sent"][m]) / med[m]) for m in S])),
                math.log1p(len(claims)) / 5.0,
                math.log1p(len(slot_values)) / 3.0,
                mean_sup,
            ])
            if al is not None:
                y.append(1.0 if is_correct(c["value"], al) else 0.0)
            rowq.append(qi)
            meta.append((q, c["slot"], c["value"]))
    X = np.array(X, float) if X else np.zeros((0, len(FEATURES)))
    return X, (np.array(y) if items else None), np.array(rowq, int), meta


def sig(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def best_threshold(p, y):
    """Claim-level F1-optimal threshold on the source (partial labels)."""
    order = np.argsort(-p)
    tp = np.cumsum(y[order]); k = np.arange(1, len(p) + 1)
    prec, rec = tp / k, tp / max(1.0, y.sum())
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    i = int(np.argmax(f1))
    return float(p[order][i])


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_within(p, y, rowq):
    """Mean per-question AUC over questions with both labels: how well p ranks
    claims INSIDE a question, which is what the composer sees. Pooled AUC also
    rewards guessing which questions have any gold-matching claim at all."""
    a = [auc(p[rowq == q], y[rowq == q]) for q in np.unique(rowq)
         if 0 < y[rowq == q].sum() < (rowq == q).sum()]
    return float(np.mean(a)) if a else float("nan")


GROUPS = {"position": ["pos_min", "pos_mean", "first_sent"],
          "contradiction": ["contradict", "margin", "slot_values"],
          "verbosity": ["sup_len", "silent_len"]}


def keep_cols(drop=()):
    gone = {f for g in drop for f in GROUPS.get(g, [g])}
    return [i for i, f in enumerate(FEATURES) if f not in gone]


class ProseScorer:
    """Fitted on the source pool's claims; applied to any other pool with no labels.
    `drop` removes feature groups (ablations); merge_contra is the
    contradiction-as-silence ablation."""

    def __init__(self, claims_by_q, train_q, src_agents, items, drop=(), merge_contra=False):
        self.cols, self.merge_contra = keep_cols(drop), merge_contra
        X, y, _, _ = featurize(claims_by_q, train_q, src_agents, items, merge_contra)
        if not len(X) or y.sum() == 0:
            raise ValueError("no labelled source claims to fit on")
        X = X[:, self.cols]
        self.w = fit_logistic(X, y)
        self.t = best_threshold(sig(X @ self.w), y)

    def predict(self, X):
        return sig(X[:, self.cols] @ self.w) if len(X) else np.zeros(0)

    def score(self, claims_by_q, qids, agents):
        X, _, rowq, meta = featurize(claims_by_q, qids, agents, merge_contra=self.merge_contra)
        p = self.predict(X)
        out = {}
        for (q, s, v), pi in zip(meta, p):
            out.setdefault(q, {})[(s, v)] = float(pi)
        return out


def split_questions(qids, seed=0):
    q = sorted(qids)
    random.Random(seed).shuffle(q)
    h = len(q) // 2
    return q[:h], q[h:]


def evaluate_transfer(claims_by_q, items, src_agents, tgt_agents, seed=0, fold=0):
    """Does p predict correct facts for agents the scorer never saw? AUC of the
    transferred p vs a support-only baseline, on held-out questions."""
    train_q, test_q = split_questions(claims_by_q, seed)
    if fold:
        train_q, test_q = test_q, train_q
    sc = ProseScorer(claims_by_q, train_q, src_agents, items)
    X, y, _, _ = featurize(claims_by_q, test_q, tgt_agents, items)
    p = sc.predict(X)
    return dict(n_claims=int(len(y)), base_rate=float(y.mean()) if len(y) else float("nan"),
                auc_scorer=auc(p, y), auc_support=auc(X[:, 1] + 1e-6 * X[:, 4], y),
                threshold=sc.t, weights=dict(zip(FEATURES, map(float, sc.w))))
