"""
Composers: the LLM step that turns scored claims into the final answer.

BAND COMPOSER (lists and prose)
  The scorer decides the claims it is confident about; the LLM rules only on
  the uncertain middle. Per question, with t the scorer's cut, in ODDS space
  (a fixed ratio works at any t; a ratio on p does not -- at t = 0.87, "2t"
  exceeds 1 and nothing is ever accepted):
      accepted   odds(p) >= ODDS * odds(t)   or  p >= P_SURE
      rejected   odds(p) <= odds(t) / ODDS   and p <  P_SURE
      undecided  everything else, capped at MAX_BAND (overflow rejected)
  ODDS, P_SURE and MAX_BAND are fixed here a priori, not tuned on test data.
  For lists, t is the smallest p the source-fitted budget kept on that
  question; for prose, the source-fitted claim-level threshold.

LONG-FORM COMPOSER (prose)
  Writes one paragraph from the agents' answers. Arms differ ONLY in what is
  added after the answers, so any gain is attributable:
      plain           nothing (= plain LLM synthesis, the baseline)
      table           the claim table: per slot, each value with p, support
                      and contradiction counts
      table_scramble  identical, p permuted across the question's claims (control)
      table_raw       the table WITHOUT p: support and contradiction counts only
                      (does the scorer add anything beyond the raw counts?)
      band            accepted / undecided / rejected facts, as above
  Every arm gets the same length instruction.
"""
import math
import random

from open_ended_aggregation.benchmarks.qampari import norm as list_norm, parse_list

ODDS, P_SURE, MAX_BAND = 3.0, 0.8, 30


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def band_split(pq, t, odds=ODDS, p_sure=P_SURE, max_band=MAX_BAND):
    """pq: {claim: p}. Returns (accepted, undecided, rejected) lists, by p desc."""
    ranked = sorted(pq, key=lambda c: -pq[c])
    lt, d = _logit(t), math.log(odds)
    acc = [c for c in ranked if _logit(pq[c]) >= lt + d or pq[c] >= p_sure]
    rej = [c for c in ranked if _logit(pq[c]) <= lt - d and pq[c] < p_sure]
    und = [c for c in ranked if c not in set(acc) | set(rej)]
    return acc, und[:max_band], und[max_band:] + rej


def list_cut(pq, kept):
    """The per-question cut for lists: smallest p among budget-kept claims."""
    if kept:
        return min(pq[c] for c in kept if c in pq)
    return max(pq.values()) if pq else 1.0


# ------------------------------------------------------------------ lists
LIST_BAND_ASK = (
    "Several assistants each answered the same question. A statistical model has "
    "already ACCEPTED some entities and REJECTED others. For each UNDECIDED entity, "
    "decide whether it is a correct answer, using the assistants' answers and your "
    "own knowledge.\n"
    "Output ONLY the undecided entities you accept, one per line, copied exactly as "
    "written below. Output nothing else. If you accept none, output NONE.\n")


def list_band_prompt(question, answers_block, acc, und):
    return (LIST_BAND_ASK + f"\nQuestion: {question}\n\n{answers_block}\n\n"
            "ACCEPTED:\n" + ("\n".join(acc) or "(none)") +
            "\n\nUNDECIDED:\n" + ("\n".join(und) or "(none)"))


def list_band_output(txt, acc, und):
    """Final list = accepted + the undecided ones the LLM accepted (others ignored)."""
    und_set = set(und)
    chosen = [list_norm(x) for x in parse_list(txt or "") if list_norm(x) in und_set]
    return list(acc) + [c for c in und if c in set(chosen)]


# ------------------------------------------------------------------ prose
PROSE_ASK = ("Several assistants answered the same question. Write the best single "
             "answer to the question, using their answers. {length}\n"
             "Output only the answer.\n")

TABLE_NOTE = (
    "\nYou are also given a claim table extracted from the answers. For each slot (a "
    "fact or sub-question) it lists every value the assistants gave, with an "
    "estimated probability that the value is correct (from a statistical model "
    "fitted on other questions; roughly calibrated), how many assistants stated it, "
    "and how many gave a different value for the same slot. Prefer high-probability "
    "values; where values for one slot conflict, give the more likely one, or say "
    "the sources disagree if the probabilities are close.\n")

TABLE_RAW_NOTE = (
    "\nYou are also given a claim table extracted from the answers. For each slot (a "
    "fact or sub-question) it lists every value the assistants gave, how many "
    "assistants stated it, and how many gave a different value for the same slot. "
    "Use it as evidence about which values are likely correct; where values for one "
    "slot conflict, give the better-supported one, or say the sources disagree.\n")

BAND_NOTE = (
    "\nA statistical model has sorted the facts stated in the answers:\n"
    "- ACCEPTED facts belong in your answer.\n"
    "- UNDECIDED facts: include them only if the answers and your knowledge support them.\n"
    "- REJECTED facts must not appear in your answer.\n")


def answers_block(responses):
    return "\n\n".join(f"--- Assistant {j} ---\n{r.strip()}"
                       for j, r in enumerate(responses, 1))


def slot_names(rec):
    return {str(s.get("id")): str(s.get("slot")) for s in rec.get("slots", [])}


def claim_table_text(claims, pq, names, n_agents, scramble_seed=None, show_p=True):
    """claims from claim_table.build; pq: {(slot, value): p}. show_p=False drops
    the probability column (and orders by support instead of p)."""
    keys = [(c["slot"], c["value"]) for c in claims]
    vals = [pq.get(k, 0.0) for k in keys]
    if scramble_seed is not None:
        random.Random(scramble_seed).shuffle(vals)
    shown = dict(zip(keys, vals))
    if show_p:
        lines = ["slot | value | p(correct) | stated by | contradicted by"]
        order = sorted(claims, key=lambda c: (c["slot"], -shown[(c["slot"], c["value"])]))
    else:
        lines = ["slot | value | stated by | contradicted by"]
        order = sorted(claims, key=lambda c: (c["slot"], -len(c["supporters"]), c["value"]))
    for c in order:
        sname = names.get(c["slot"], c["slot"].replace("NEW:", "").strip())
        pcol = f" | {shown[(c['slot'], c['value'])]:.2f}" if show_p else ""
        lines.append(f"{sname} | {c.get('display', c['value'])}{pcol} | "
                     f"{len(c['supporters'])} of {n_agents} | {len(c['contradict'])}")
    return "\n".join(lines)


def band_text(claims, pq, names, t):
    fact = {(c["slot"], c["value"]): f"{names.get(c['slot'], c['slot'])}: "
                                      f"{c.get('display', c['value'])}" for c in claims}
    acc, und, rej = band_split({k: pq.get(k, 0.0) for k in fact}, t)
    return ("ACCEPTED:\n" + ("\n".join(fact[k] for k in acc) or "(none)") +
            "\n\nUNDECIDED:\n" + ("\n".join(fact[k] for k in und) or "(none)") +
            "\n\nREJECTED:\n" + ("\n".join(fact[k] for k in rej) or "(none)"))


def prose_prompt(arm, question, responses, max_words, claims=None, pq=None,
                 names=None, t=None, seed=None):
    """One composer prompt. `arm` in plain | table | table_scramble | table_raw | band."""
    head = PROSE_ASK.format(length=f"Write one paragraph of at most {max_words} words.")
    body = f"\nQuestion: {question}\n\n{answers_block(responses)}\n"
    if arm == "plain":
        return head + body
    n = len(responses)
    if arm == "table":
        return head + body + TABLE_NOTE + "\n" + claim_table_text(claims, pq, names, n)
    if arm == "table_scramble":
        return head + body + TABLE_NOTE + "\n" + claim_table_text(claims, pq, names, n,
                                                                  scramble_seed=seed)
    if arm == "table_raw":
        return head + body + TABLE_RAW_NOTE + "\n" + claim_table_text(claims, pq, names, n,
                                                                      show_p=False)
    if arm == "band":
        return head + body + BAND_NOTE + "\n" + band_text(claims, pq, names, t)
    raise ValueError(arm)
