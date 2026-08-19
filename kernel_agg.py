"""
Kernel-Weighted Aggregation (KWA) for open-ended LLM answers.
============================================================

Motivation
----------
Ai, Pan, Simchi-Levi, Tambe & Xu (ICML 2026), "Beyond Majority Voting: LLM
Aggregation by Leveraging Higher-Order Information" (arXiv:2510.01499), derive a
Bayes-optimal aggregator for K-way multiple choice:

    Optimal Weight (OW):   argmax_s  sum_j  w_j * 1{a_j = s},
                           w_j = log( (K-1) x_j / (1 - x_j) ) = sigma_K^{-1}(x_j)

Their derivation needs TWO facts, both manufactured by randomly shuffling the
answer labels:

    (1)  P(agent j says s  | truth = s )  =  x_j
    (2)  P(agent j says s' | truth = s )  =  (1 - x_j)/(K - 1)   for EVERY s' != s

Fact (2) -- every wrong answer equally likely -- is what lets a constant factor
out of the likelihood product, leaving `weight x indicator`, i.e. a vote.

Their own future-work section asks: "How to derive optimal weights for
open-ended questions?"  Open-ended generation destroys fact (2): there is no
label set to shuffle, and "Sydney" is a far likelier error than "photosynthesis"
when the truth is "Canberra".

The generalisation implemented here
-----------------------------------
Replace the exact-match indicator with a similarity kernel:

    log P(agent j answers a | truth = s)  =  alpha_j + beta_j * sim(a, s)

beta_j is how sharply agent j concentrates its probability mass near the truth
(a strong agent has large beta and answers close to correct; a weak agent
sprays).  Since alpha_j does not depend on s, the Bayes rule is

    KWA:   argmax_s  sum_j  beta_j * sim(a_j, s)

Exact reduction to OW
---------------------
Set sim(a, s) = 1{a = s}.  Then P(a | truth = s) ∝ exp(alpha_j + beta_j 1{a=s}),
and normalising over K options gives

    P(correct) = e^(a+b) / [ e^(a+b) + (K-1) e^a ] = e^b / (e^b + K - 1) = sigma_K(beta_j)

which is exactly the paper's own sigma_K parametrisation (their Section D.1), and
the rule collapses to OW.  Multiple choice is the special case where similarity
happens to be binary.  `test_reduction()` checks this numerically.

Estimating beta without labels
------------------------------
EM with the truth as a latent variable, ranging over the candidate pool:

    E-step:  gamma_q(s)  ∝  prod_j P(a_j | truth = s, beta)
    M-step:  maximise  sum_q sum_s gamma_q(s) sum_j log P(a_j | s, beta_j)

The M-step separates across agents, and each per-agent objective

    beta_j * S[a_j, s]  -  log sum_c exp(beta_j * S[c, s])

is concave in beta_j (linear minus log-sum-exp), so each is a reliable 1-D
concave maximisation.  This is Dawid-Skene generalised from a confusion matrix
to a similarity kernel.
"""

import numpy as np
from scipy.optimize import minimize_scalar

# --------------------------------------------------------------------------
# generative model for a question
# --------------------------------------------------------------------------

def make_pool(rng, d=64, n_para=2, n_near=3, n_far=4,
              t_para=0.975, t_near=0.75, t_far=0.30, jitter=0.012):
    """Build one question's candidate answer pool as unit vectors in R^d.

    Layout (index 0 is always the truth):
        0                     the true answer
        1 .. n_para           PARAPHRASES of the truth  (counted as CORRECT)
        .. n_near             plausible-but-wrong, semantically near
        .. n_far              implausible, semantically far

    Paraphrases matter: they are why exact-match majority voting struggles on
    open-ended output.  Three agents saying "Canberra", "Canberra." and "The
    capital is Canberra" have their votes SPLIT by exact match, while a single
    wrong answer can win a plurality.  A kernel pools them.
    """
    u0 = rng.normal(size=d); u0 /= np.linalg.norm(u0)
    targets = ([1.0]
               + [t_para] * n_para
               + [t_near] * n_near
               + [t_far] * n_far)
    U = [u0]
    for t in targets[1:]:
        t = float(np.clip(t + rng.normal(0, jitter), -0.99, 0.99))
        v = rng.normal(size=d)
        v -= v.dot(u0) * u0                 # orthogonalise
        v /= np.linalg.norm(v)
        U.append(t * u0 + np.sqrt(1 - t**2) * v)
    U = np.array(U)
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def sample_answers(S_to_truth, betas, rng, family=None, family_pull=0.0):
    """Each agent j picks candidate c with probability ∝ exp(beta_j * sim(c, truth)).

    `family` / `family_pull` inject the correlated-error structure that the
    paper's Assumption 2.2 forbids: agents sharing a family get an extra shared
    logit bump toward one particular distractor, so they err TOGETHER and in the
    SAME DIRECTION.  This is the same-lineage correlation real ensembles have
    (shared pretraining data, tokenizer, architecture).
    """
    N = len(betas)
    C = len(S_to_truth)
    out = np.empty(N, dtype=int)
    bump = np.zeros((N, C))
    if family is not None and family_pull > 0:
        for f in set(family):
            members = [i for i in range(N) if family[i] == f]
            target = rng.integers(1, C)           # a shared attractive wrong answer
            for i in members:
                bump[i, target] = family_pull
    for j in range(N):
        logits = betas[j] * S_to_truth + bump[j]
        p = np.exp(logits - logits.max()); p /= p.sum()
        out[j] = rng.choice(C, p=p)
    return out


def make_dataset(M, betas, rng, pool_kw=None, family=None, family_pull=0.0,
                 curvature=1.0):
    """Generate M questions.  Returns (answers[M,N], sims[M,C,C], truth_mask[C]).

    `curvature` != 1 MISSPECIFIES the model: the true log-probability becomes
    beta * sim^curvature rather than beta * sim, so the linear-in-similarity
    assumption is wrong.  Used to test robustness.
    """
    pool_kw = pool_kw or {}
    N = len(betas)
    A, Ss = [], []
    for _ in range(M):
        U = make_pool(rng, **pool_kw)
        S = U @ U.T
        s_truth = S[0].copy()
        drive = np.sign(s_truth) * np.abs(s_truth) ** curvature
        A.append(sample_answers(drive, betas, rng, family, family_pull))
        Ss.append(S)
    return np.array(A), np.array(Ss)


def correct_mask(S, thresh=0.95, n_correct=None):
    """Which candidates count as a correct answer (the truth or a paraphrase).

    Two modes. `thresh` marks anything similar enough to the truth as correct --
    fine when paraphrases are far above distractors. On REAL data they are not
    (measured gap ~0.05), so `n_correct` instead marks the first n_correct pool
    entries correct BY CONSTRUCTION, which is the honest version.
    """
    if n_correct is not None:
        m = np.zeros(S.shape[0], dtype=bool)
        m[:n_correct] = True
        return m
    return S[0] >= thresh


# --------------------------------------------------------------------------
# aggregators
# --------------------------------------------------------------------------

def _argmax_random_tie(scores, rng):
    best = scores.max()
    idx = np.flatnonzero(scores >= best - 1e-9)
    return int(rng.choice(idx))


def agg_majority_exact(ans, S, rng, **kw):
    """Plain majority vote on exact answer identity."""
    C = S.shape[0]
    v = np.bincount(ans, minlength=C).astype(float)
    return _argmax_random_tie(v, rng)


def agg_majority_cluster(ans, S, rng, tau=0.90, **kw):
    """Universal-Self-Consistency style: single-linkage cluster the OBSERVED
    answers at similarity tau, majority vote over clusters, report the cluster's
    most-voted member.  This is the strong incumbent baseline for open-ended."""
    obs = np.unique(ans)
    parent = {c: c for c in obs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, ci in enumerate(obs):
        for cj in obs[i+1:]:
            if S[ci, cj] >= tau:
                parent[find(ci)] = find(cj)
    counts, members = {}, {}
    for a in ans:
        r = find(a)
        counts[r] = counts.get(r, 0) + 1
        members.setdefault(r, []).append(a)
    roots = list(counts)
    best_root = roots[_argmax_random_tie(np.array([counts[r] for r in roots], float), rng)]
    mem, cnt = np.unique(members[best_root], return_counts=True)
    return int(mem[_argmax_random_tie(cnt.astype(float), rng)])


def agg_ow_exact(ans, S, rng, weights=None, **kw):
    """Optimal Weight on exact match -- the paper's Algorithm 1, unchanged."""
    C = S.shape[0]
    sc = np.zeros(C)
    for j, a in enumerate(ans):
        sc[a] += weights[j]
    return _argmax_random_tie(sc, rng)


def agg_kernel(ans, S, rng, betas=None, support=None, **kw):
    """Kernel-Weighted Aggregation:  argmax_s sum_j beta_j * sim(a_j, s).

    `support` restricts the candidates considered.  None = the whole pool
    (oracle); 'observed' = only answers some agent actually produced, which is
    the honest deployable setting.
    """
    C = S.shape[0]
    sc = np.zeros(C)
    for j, a in enumerate(ans):
        sc += betas[j] * S[a]
    if support == "observed":
        mask = np.full(C, -np.inf)
        obs = np.unique(ans)
        mask[obs] = 0.0
        sc = sc + mask
    return _argmax_random_tie(sc, rng)


# --------------------------------------------------------------------------
# the paper's label-free methods, applied to exact-match answers
# --------------------------------------------------------------------------

def second_order_table(A, C):
    """P(A_i = c_k | A_j = c_l) from co-occurrence counts. Answer INDICES are not
    comparable across questions here (each question has its own pool), so we work
    in the only cross-question-stable coordinate available: 'agent i and agent j
    gave the same answer' vs not.  That is the K=2 collapse of the paper's table
    and is what its machinery reduces to when labels carry no shared meaning."""
    N = A.shape[1]
    agree = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                agree[i, j] = (A[:, i] == A[:, j]).mean()
    return agree


def estimate_acc_from_agreement(agree, K_eff):
    """OW-L: solve for accuracies by matching observed pairwise agreement to
        agree(i,j) = x_i x_j + (1-x_i)(1-x_j)/(K_eff - 1)
    """
    from scipy.optimize import least_squares
    N = agree.shape[0]
    idx = [(i, j) for i in range(N) for j in range(N) if i != j]
    obs = np.array([agree[i, j] for i, j in idx])

    def resid(x):
        pred = np.array([x[i]*x[j] + (1-x[i])*(1-x[j])/(K_eff-1) for i, j in idx])
        return pred - obs

    best = None
    for st in (0.7, 0.5, 0.9, 1.0/K_eff + 0.05):
        r = least_squares(resid, np.full(N, st),
                          bounds=(np.full(N, 1.0/K_eff), np.full(N, 1 - 1e-9)))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


# --------------------------------------------------------------------------
# EM: recover beta with NO labels
# --------------------------------------------------------------------------

def _agent_logp(beta, S, a_idx, support_mask):
    """log P(a_q | truth = s) for every question q and candidate s, vectorised.

    returns [M, C]:   beta*S[q, a_q, s]  -  logsumexp_{c in support} beta*S[q, c, s]
    """
    M, C, _ = S.shape
    lin = beta * S[np.arange(M), a_idx, :]                  # [M, C]
    Z = np.where(support_mask[:, :, None], beta * S, -np.inf)   # [M, C(c), C(s)]
    lse = np.logaddexp.reduce(Z, axis=1)                    # [M, C]
    return lin - lse


def _loglik_agent(beta, S, a_idx, gamma, support_mask):
    return float((gamma * _agent_logp(beta, S, a_idx, support_mask)).sum())


def em_estimate_beta(A, S, n_iter=40, support="observed", beta0=2.0,
                     bounds=(0.01, 30.0), tol=1e-3, verbose=False):
    """Label-free EM for beta.  Returns (beta_hat, history)."""
    M, N = A.shape
    C = S.shape[1]
    support_mask = np.zeros((M, C), dtype=bool)
    if support == "observed":
        support_mask[np.arange(M)[:, None], A] = True
    else:
        support_mask[:] = True

    beta = np.full(N, float(beta0))
    hist = []
    for it in range(n_iter):
        # ---- E-step: posterior over which candidate is the truth
        logp = np.zeros((M, C))
        for j in range(N):
            logp += _agent_logp(beta[j], S, A[:, j], support_mask)
        logp = np.where(support_mask, logp, -np.inf)   # truth must lie in the support
        logp -= logp.max(axis=1, keepdims=True)
        g = np.exp(logp)
        gamma = g / g.sum(axis=1, keepdims=True)

        # ---- M-step: one concave 1-D problem per agent
        new = np.empty(N)
        for j in range(N):
            f = lambda b: -_loglik_agent(b, S, A[:, j], gamma, support_mask)
            r = minimize_scalar(f, bounds=bounds, method="bounded",
                                options={"xatol": 1e-3})
            new[j] = r.x
        shift = np.abs(new - beta).max()
        beta = new
        hist.append(beta.copy())
        if verbose:
            print(f"   EM iter {it:2d}  beta = {np.round(beta,3)}  shift {shift:.4f}")
        if shift < tol:
            break
    return beta, hist


# --------------------------------------------------------------------------
# correctness check: KWA must equal OW when the kernel is exact match
# --------------------------------------------------------------------------

def test_reduction(seed=0, K=4, N=5, trials=4000):
    """With sim = identity, KWA(beta) must pick exactly what OW(sigma_K(beta)) picks."""
    rng = np.random.default_rng(seed)
    S = np.eye(K)
    betas = rng.uniform(0.2, 4.0, size=N)
    x = np.exp(betas) / (K - 1 + np.exp(betas))          # sigma_K(beta)
    w = np.log((K - 1) * x / (1 - x))                    # sigma_K^{-1}(x) -- OW weight
    assert np.allclose(w, betas), "sigma_K inverse round-trip failed"
    same = 0
    for _ in range(trials):
        ans = rng.integers(0, K, size=N)
        r1 = np.random.default_rng(123)
        r2 = np.random.default_rng(123)
        same += (agg_kernel(ans, S, r1, betas=betas) ==
                 agg_ow_exact(ans, S, r2, weights=w))
    return same / trials
