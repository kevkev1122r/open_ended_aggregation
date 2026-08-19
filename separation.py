"""How well can similarity ACTUALLY tell a real error from a control? (AUC)

AUC answers a plain question: show the method one real model error and one
control answer. How often does it correctly pick out the real one, using
similarity alone?
    0.50 = coin flip, the signal is useless
    1.00 = perfect
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from regression import load, encode, build_rows, ENCODERS
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=f"{HERE}/results/figures"
BLUE,GREY="#1a73e8","#9aa0a6"

def auc(pos, neg):
    a=np.concatenate([pos,neg]); r=a.argsort().argsort().astype(float)+1
    n1,n0=len(pos),len(neg)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)

fig,axes=plt.subplots(1,3,figsize=(15,4.6))
print(f"  {'dataset':<14}{'AUC (hard controls)':>22}{'plain meaning':>44}")
print("  "+"-"*80)
for ax,ds in zip(axes,["TruthfulQA","HaluEval-QA","SciQ"]):
    t,w=load(ds); Et,Ew=encode(ENCODERS[0],t),encode(ENCODERS[0],w)
    sim,y,cl=build_rows(Et,Ew,hard=True)
    pos,neg=sim[y==1],sim[y==0]
    A=auc(pos,neg)
    print(f"  {ds:<14}{A:>22.3f}{f'right {100*A:.0f}% of the time':>44}")
    bins=np.linspace(min(sim.min(),-0.1),sim.max(),55)
    ax.hist(neg,bins=bins,density=True,color=GREY,alpha=.7,label="control answers")
    ax.hist(pos,bins=bins,density=True,color=BLUE,alpha=.7,label="REAL model errors")
    ax.set_title(f"{ds}\nAUC = {A:.2f}  →  picks right {100*A:.0f}% of the time",
                 fontsize=11.5,fontweight="bold")
    ax.set_xlabel("similarity to the truth"); ax.grid(alpha=.2)
    ax.legend(fontsize=8.5)
axes[0].set_ylabel("how common")
axes[0].text(.5,.55,"barely overlap\n= easy to tell apart",transform=axes[0].transAxes,
             ha="center",fontsize=10,color="#0b8043",fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35",fc="#e6f4ea",ec="#0b8043"))
for ax in axes[1:]:
    ax.text(.62,.55,"heavy overlap\n= hard to tell apart",transform=ax.transAxes,
            ha="center",fontsize=10,color="#b3261e",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",fc="#fce8e6",ec="#b3261e"))
fig.suptitle("The same method, three tasks: how much can similarity actually tell you?",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig(f"{FIG}/R5_separation.png",dpi=150,facecolor="white")
print("\n  -> R5_separation.png")
