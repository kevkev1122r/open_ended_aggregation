"""Where the dots ACTUALLY come from: two piles of answers, sliced by similarity."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=f"{HERE}/results/figures"
BLUE,GREY,RED,GREEN="#1a73e8","#9aa0a6","#b3261e","#0b8043"

from sentence_transformers import SentenceTransformer
m=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
d=pd.read_parquet(f"{HERE}/data/halueval_qa.parquet").dropna(
   subset=["right_answer","hallucinated_answer"]).reset_index(drop=True).iloc[:3000]
E=lambda t:m.encode(list(t),batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
Et,Ew=E(d["right_answer"]),E(d["hallucinated_answer"])
pos=np.einsum("ij,ij->i",Ew,Et)                       # 3000 REAL model errors
sim=Et@Et.T; np.fill_diagonal(sim,-np.inf); order=np.argsort(-sim,axis=1)
K=8
idx=order[:,:K]
ctl=np.stack([np.einsum("ij,ij->i",Ew[idx[:,j]],Et) for j in range(K)],1).ravel()   # 24000 comparison

print(f"PILE 1  real model errors      : {len(pos):,} answers")
print(f"PILE 2  comparison answers      : {len(ctl):,} answers   ({K} per question)")
print(f"        ratio of pile sizes     : {len(ctl)//len(pos)}x more comparison answers\n")

e=np.quantile(np.concatenate([pos,ctl]),np.linspace(0.02,0.98,11))
print(f"  {'slice of similarity':>26}{'REAL errors':>13}{'comparison':>12}{'ratio':>9}{'x8':>8}")
print("  "+"-"*68)
rows=[]
for i,(a,b) in enumerate(zip(e[:-1],e[1:])):
    p=int(((pos>=a)&(pos<b)).sum()); q=int(((ctl>=a)&(ctl<b)).sum())
    if p>=5 and q>=5:
        star=" <-- the one I showed you" if abs((a+b)/2-0.29)<0.02 else ""
        print(f"  {a:>10.2f} to {b:<5.2f}{'':>6}{p:>10}{q:>12}{p/q:>9.3f}{p/q*K:>8.2f}{star}")
        rows.append(((a+b)/2,p,q))
print(f"""
  The last column is the one that matters. Because there are {K}x more comparison
  answers overall, a ratio of 1/{K} = {1/K:.3f} would mean 'no signal at all'. Multiplying
  by {K} rescales it so that 1.00 = no signal, above 1 = more likely to be a real error.
""")

c=np.array([r[0] for r in rows]); lo=np.log([r[1]/r[2]*K for r in rows])
hi_i=int(np.argmin(np.abs(c-0.29)))

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,8.6),height_ratios=[1.25,1],sharex=True)
bins=np.linspace(-0.15,0.75,70)
ax1.hist(ctl,bins=bins,color=GREY,alpha=.75,label=f"PILE 2 — comparison answers ({len(ctl):,})",
         weights=np.ones_like(ctl)/K)
ax1.hist(pos,bins=bins,color=BLUE,alpha=.75,label=f"PILE 1 — REAL model errors ({len(pos):,})")
for x in e: ax1.axvline(x,color="k",ls="--",lw=.8,alpha=.45)
lo_e,hi_e=e[hi_i],e[hi_i+1]
ax1.axvspan(lo_e,hi_e,color="#fbbc04",alpha=.35,zorder=0)
ax1.set_ylabel("how many answers\n(grey scaled ÷8 to compare fairly)",fontsize=10)
ax1.legend(fontsize=10,loc="upper right")
ax1.set_title("Every answer, dropped into a slice by how similar it is to the right answer\n"
              "dashed lines = the 10 slices.  0.29 is nothing special — it is just one slice.",
              fontsize=12,fontweight="bold")
ax1.annotate(f"this slice:\n{rows[hi_i][1]} real errors\n{rows[hi_i][2]} comparison",
             xy=((lo_e+hi_e)/2,ax1.get_ylim()[1]*.55),xytext=(0.47,ax1.get_ylim()[1]*.75),
             fontsize=10,fontweight="bold",
             arrowprops=dict(arrowstyle="->",lw=1.6,color="#f9ab00"),
             bbox=dict(boxstyle="round,pad=0.4",fc="#fef7e0",ec="#f9ab00"))

ax2.plot(c,lo,"o",ms=11,color=BLUE)
ax2.plot(c[hi_i],lo[hi_i],"o",ms=18,mfc="none",mec="#f9ab00",mew=3)
ax2.axhline(0,color=GREY,ls=":",lw=1.2)
ax2.text(0.60,0.03,"this height = 'no signal'",fontsize=9,color=GREY,style="italic")
for x in e: ax2.axvline(x,color="k",ls="--",lw=.8,alpha=.45)
ax2.set_xlabel("similarity to the right answer  →",fontsize=11)
ax2.set_ylabel("one dot per slice\n(log of the last column)",fontsize=10)
ax2.grid(alpha=.25)
ax2.set_title("Each slice above becomes exactly one dot down here",fontsize=11.5,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/T3_where_dots_come_from.png",dpi=150,facecolor="white")
print("  -> T3_where_dots_come_from.png")
