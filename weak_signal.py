"""Is a 64%-AUC signal too weak to aggregate on?

The whole premise of aggregation is that MANY weak signals combine into one
strong decision. So the per-answer AUC being 0.64 does not by itself sink the
method -- what matters is what happens after pooling N agents. Test it directly:
dial the simulated signal down until it matches the WEAKEST real measurement,
then run the aggregators.
"""
import functools, numpy as np, kernel_agg as ka, experiments as ex

REAL = dict(n_para=2, n_near=3, n_far=4, t_para=0.626, t_near=0.580, t_far=0.058, jitter=0.10)
NCORR = 3
M = 2000

def auc(pos, neg):
    a=np.concatenate([pos,neg]); r=a.argsort().argsort().astype(float)+1
    n1,n0=len(pos),len(neg); return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)

def measure_auc(A, S):
    """Per-answer AUC, exactly as measured on the real data."""
    rng=np.random.default_rng(0); pos=[]; neg=[]
    for q in range(len(A)):
        for j in range(A.shape[1]):
            pos.append(S[q][0, A[q,j]])
        q2=rng.integers(0,len(A))
        for j in range(A.shape[1]):
            neg.append(S[q][0, A[q2,j]])       # another question's answer, this truth
    return auc(np.array(pos), np.array(neg))

orig = ka.correct_mask
ka.correct_mask = functools.partial(orig, n_correct=NCORR)
try:
    print("  dialling the signal down to match the real measurements...\n")
    print(f"  {'beta scale':>11}{'per-answer AUC':>17}{'MV-exact':>11}{'MV-cluster*':>13}"
          f"{'KWA-EM':>9}{'gain':>8}")
    print("  "+"-"*70)
    for scale in (8.0, 5.0, 3.0, 2.0, 1.4, 1.0):
        betas=[scale*b/8.0*8 for b in (1.0,0.85,0.7,0.55,0.42)]  # keep spread, scale level
        betas=[scale*x for x in (1.0,0.85,0.70,0.55,0.42)]
        accs={}
        aucs=[]
        for s in range(3):
            rng=np.random.default_rng(2000+s)
            A,S=ka.make_dataset(M,betas,rng,pool_kw=REAL)
            aucs.append(measure_auc(A,S))
            bh,_=ka.em_estimate_beta(A,S,support="observed")
            # tuned cluster baseline
            best=-1
            for tau in (0.4,0.5,0.6,0.65,0.7):
                r2=np.random.default_rng(s)
                a=np.mean([bool(ka.correct_mask(S[q])[ka.agg_majority_cluster(A[q],S[q],r2,tau=tau)])
                           for q in range(M)])
                best=max(best,a)
            r2=np.random.default_rng(s)
            mv=np.mean([bool(ka.correct_mask(S[q])[ka.agg_majority_exact(A[q],S[q],r2)]) for q in range(M)])
            r2=np.random.default_rng(s)
            kw=np.mean([bool(ka.correct_mask(S[q])[ka.agg_kernel(A[q],S[q],r2,betas=bh,support="observed")])
                        for q in range(M)])
            accs.setdefault("mv",[]).append(100*mv); accs.setdefault("cl",[]).append(100*best)
            accs.setdefault("kw",[]).append(100*kw)
        mv,cl,kw=(np.mean(accs[k]) for k in ("mv","cl","kw"))
        tag=""
        a=np.mean(aucs)
        if abs(a-0.64)<0.03: tag="  <-- matches HaluEval / SciQ"
        if abs(a-0.845)<0.03: tag="  <-- matches TruthfulQA"
        print(f"  {scale:>11.1f}{a:>17.3f}{mv:>11.2f}{cl:>13.2f}{kw:>9.2f}{kw-cl:>+8.2f}{tag}")
finally:
    ka.correct_mask = orig
print("""
  * MV-cluster is tuned to its best threshold at every row, so it is not a strawman.
  The question is whether the KWA gain survives as the per-answer signal weakens.""")
