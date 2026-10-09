"""
Published multi-response baselines: USC, Mixture-of-Agents, LLM-Blender.

PAPER-EXACTNESS (memory rule: never substitute an adapted variant)
  USC and MoA prompts were checked on 23 Sept 2026 against the paper / official
  repository (see the VERIFIED notes); LLM-Blender's API against its README
  (ranker: the documented PairRM checkpoint). Deviations the setting forces are listed per method and must
  be reported next to the result.

USC  Universal Self-Consistency, Chen et al. 2023, arXiv:2311.17311.
     The LLM is shown all responses and selects the most consistent one; the
     output IS that response (a selector, not a merger).
     Forced deviation: none for selection. Responses come from different agents
     instead of samples of one model (the setting of this project).

MoA  Mixture-of-Agents, Wang et al. 2024, arXiv:2406.04692.
     Proposers answer; an aggregator gets the "Aggregate-and-Synthesize"
     system prompt containing all responses, and the user query as the user
     message. The paper's default stacks 3 layers of proposers.
     Forced deviation (reported): layers=1 re-uses the cached agent answers as
     the single proposer layer, so this is the MoA aggregator over one layer.
     moa_layer() builds further layers but they re-query the agents, several of
     which are Azure deployments billed to a personal card; not run by default.

LLM-Blender  Jiang et al. 2023, arXiv:2306.02561.
     PairRanker (pairwise ranker) picks the top-K responses; GenFuser (a
     Flan-T5 fuser) generates the final answer from them. Runs LOCALLY, no API
     cost, but needs the `llm-blender` package and the released model weights
     (several GB). Neither is installed; installing and downloading need
     Kevin's explicit approval, so this raises until then.
"""
import re

# ------------------------------------------------------------------ USC
# VERIFIED 23 Sept 2026 against arXiv:2311.17311 (HTML version, USC prompt):
# wording and "Response 0.." enumeration match. The paper shows the prompt in a
# figure, so line breaks follow the figure's layout (one per line) as rendered.
USC_PROMPT = ("I have generated the following responses to the question: {question}\n"
              "{responses}\n"
              "Evaluate these responses. Select the most consistent response based on "
              'majority consensus. Start your answer with "The most consistent response '
              'is Response X" (without quotes).')


def usc_prompt(question, responses):
    body = "\n".join(f"Response {i}: {r}" for i, r in enumerate(responses))
    return USC_PROMPT.format(question=question, responses=body)


def usc_choice(txt, n):
    """Index of the selected response, or None if unparseable / out of range."""
    m = re.search(r"most consistent response is Response\s*(\d+)", txt or "", re.I)
    if not m:
        return None
    i = int(m.group(1))
    return i if 0 <= i < n else None


# ------------------------------------------------------------------ MoA
# VERIFIED 23 Sept 2026 verbatim against the official repository
# (togethercomputer/MoA, utils.py, inject_references_to_messages): same text,
# references appended as "\n{i+1}. {reference}" (unstripped), sent as the
# SYSTEM message ahead of the user query.
MOA_SYSTEM = (
    "You have been provided with a set of responses from various open-source models "
    "to the latest user query. Your task is to synthesize these responses into a "
    "single, high-quality response. It is crucial to critically evaluate the "
    "information provided in these responses, recognizing that some of it may be "
    "biased or incorrect. Your response should not simply replicate the given "
    "answers but should offer a refined, accurate, and comprehensive reply to the "
    "instruction. Ensure your response is well-structured, coherent, and adheres to "
    "the highest standards of accuracy and reliability.\n\n"
    "Responses from models:")


def moa_messages(user_query, responses):
    """(system, user) for the MoA aggregator."""
    system = MOA_SYSTEM
    for i, r in enumerate(responses):
        system += f"\n{i+1}. {r}"                     # exactly as the official repo
    return system, user_query


def moa_layer(proposers, user_query, prev_responses):
    """One further MoA layer: every proposer answers the query with the previous
    layer's responses in its system prompt. `proposers` are callables
    (msg, system=...) -> (text, meta). Not used by default (see module note)."""
    system, user = moa_messages(user_query, prev_responses)
    return [p(user, system=system)[0] or "" for p in proposers]


# ------------------------------------------------------------------ LLM-Blender
class Blender:
    """PairRanker + GenFuser, run locally. Raises until installed and approved.

    API VERIFIED 23 Sept 2026 against the official README (yuchenlin/LLM-Blender):
    Blender(), loadranker, loadfuser, rank(..., return_scores=False, batch_size=1),
    get_topk_candidates_from_ranks(ranks, candidates, top_k=3), fuse(...).
    The README documents only the ranker checkpoint "llm-blender/PairRM"; the
    ACL-2023 paper's original PairRanker checkpoint is not listed there. Report
    this baseline as "LLM-Blender with the official PairRM ranker + GenFuser".
    """

    def __init__(self, ranker="llm-blender/PairRM", fuser="llm-blender/gen_fuser_3b",
                 top_k=3):
        try:
            import llm_blender  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "LLM-Blender is not installed. Installing the package and downloading "
                "the ranker and fuser weights (several GB) needs Kevin's approval first.")
        import llm_blender
        self.top_k = top_k
        self.b = llm_blender.Blender()
        self.b.loadranker(ranker)
        self.b.loadfuser(fuser)

    def __call__(self, instruction, candidates):
        from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
        ranks = self.b.rank([instruction], [candidates], return_scores=False, batch_size=1)
        topk = get_topk_candidates_from_ranks(ranks, [candidates], top_k=self.top_k)
        return self.b.fuse([instruction], topk, batch_size=1)[0]
