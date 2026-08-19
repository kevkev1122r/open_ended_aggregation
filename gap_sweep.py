"""What ACTUALLY governs aggregation accuracy: the within-question gap.

I conflated two different measurements last time. Setting them straight:

  (A) The regression's AUC 0.64 asks: given ONE answer, is it a real model error
      or a control drawn from another question? That measures how well the
      log-odds MODEL fits. It is not what the aggregator does.

  (B) The aggregator asks: given N answers to ONE question, which candidate is
      the truth? That is governed by the gap between correct and plausible-wrong
      candidates INSIDE a single question's pool -- measured on real data at
      0.626 vs 0.580, a gap of 0.046.

(A) does not translate into (B). This sweeps (B), the quantity that matters,
around the real measured value.
"""
import functools, numpy as np, kernel_agg as ka, experiments as ex
M, NCORR = 2000, 3
BET = [8.0, 6.0, 4.0, 2.5, 1.5]
orig = ka.correct_mask
ka.correct_mask = functools.partial(orig, n_correct=NCORR)

def within_auc(S_list):
    """Can similarity-to-truth separate CORRECT from WRONG candidates in a pool?"""
    pos, neg = [], []
    for S in S_list[:400]:
        pos += list(S[0, 1:NCORR]); neg += list(S[0, NCORR:])
    pos, neg = np.array(pos), np.array(neg)
    a=np.concatenate([pos,neg]); r=a.argsort().argsort().astype(float)+1
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

try:
    print(f"  {'gap':>6}{'within-Q AUC':>15}{'MV-exact':>11}{'MV-cluster*':>13}{'KWA-EM':>9}{'gain':>8}")
    print("  "+"-"*64)
    for gap in (0.40, 0.20, 0.10, 0.046, 0.02):
        pk = dict(n_para=2, n_near=3, n_far=4, t_para=0.580+gap, t_near=0.580,
                  t_far=0.058, jitter=0.10)
        mv=cl=kw=0.0; aucs=[]
        for s in range(3):
            rng=np.random.default_rng(3000+s)
            A,S=ka.make_dataset(M,BET,rng,pool_kw=pk)
            aucs.append(within_auc(S))
            bh,_=ka.em_estimate_beta(A,S,support="observed")
            best=-1
            for tau in (0.4,0.5,0.55,0.6,0.65,0.7,0.8):
                r2=np.random.default_rng(s)
                best=max(best,np.mean([bool(ka.correct_mask(S[q])[
                    ka.agg_majority_cluster(A[q],S[q],r2,tau=tau)]) for q in range(M)]))
            r2=np.random.default_rng(s)
            mv+=np.mean([bool(ka.correct_mask(S[q])[ka.agg_majority_exact(A[q],S[q],r2)]) for q in range(M)])
            r2=np.random.default_rng(s)
            kw+=np.mean([bool(ka.correct_mask(S[q])[ka.agg_kernel(A[q],S[q],r2,betas=bh,support="observed")])
                         for q in range(M)])
            cl+=best
        mv,cl,kw=100*mv/3,100*cl/3,100*kw/3
        tag="   <-- REAL measured gap" if abs(gap-0.046)<1e-6 else ""
        print(f"  {gap:>6.3f}{np.mean(aucs):>15.3f}{mv:>11.2f}{cl:>13.2f}{kw:>9.2f}{kw-cl:>+8.2f}{tag}")
finally:
    ka.correct_mask = orig
print("\n  * cluster baseline tuned to its best threshold on every row.")
