"""Fitted logistic curves: linear vs quadratic, per dataset. Shows the DIRECTION of bend."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
from regression import load, encode, build_rows, ENCODERS, K
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=f"{HERE}/results/figures"
BLUE,RED,GREEN="#1a73e8","#b3261e","#0b8043"

fig,axes=plt.subplots(1,3,figsize=(15.5,5.0))
for ax,ds in zip(axes,["HaluEval-QA","TruthfulQA","SciQ"]):
    t,w=load(ds); Et,Ew=encode(ENCODERS[0],t),encode(ENCODERS[0],w)
    sim,y,cl=build_rows(Et,Ew,hard=True)
    X1=sm.add_constant(sim); X2=sm.add_constant(np.column_stack([sim,sim**2]))
    m1=sm.Logit(y,X1).fit(disp=0); m2=sm.Logit(y,X2).fit(disp=0)
    # binned observed log-odds for reference
    e=np.quantile(sim,np.linspace(.02,.98,13)); c=[];lo=[]
    for a,b in zip(e[:-1],e[1:]):
        msk=(sim>=a)&(sim<b)
        p=int(y[msk].sum()); q=int((~y[msk].astype(bool)).sum())
        if p>=5 and q>=5: c.append((a+b)/2); lo.append(np.log(p/q))
    c=np.array(c); lo=np.array(lo)
    xs=np.linspace(np.percentile(sim,1),np.percentile(sim,99),300)
    l1=m1.params[0]+m1.params[1]*xs
    l2=m2.params[0]+m2.params[1]*xs+m2.params[2]*xs**2
    ax.plot(c,lo,"o",ms=9,color=BLUE,label="observed (binned, for reference)",zorder=5)
    ax.plot(xs,l1,"-",lw=2.6,color=RED,label="fitted STRAIGHT line")
    ax.plot(xs,l2,"-",lw=2.6,color=GREEN,label="fitted CURVE")
    bend=m2.params[2]
    ax.set_title(f"{ds}\nbend = {bend:+.1f}  →  "
                 f"{'curves UP (accelerates)' if bend>0 else 'curves DOWN (saturates)'}",
                 fontsize=11.5,fontweight="bold",
                 color=GREEN if bend>0 else "#e8871a")
    ax.set_xlabel("similarity to the truth →"); ax.grid(alpha=.25)
    ax.legend(fontsize=8.5,loc="upper left")
    ax.text(.97,.04,f"variance explained\nstraight {m1.prsquared:.3f} → curve {m2.prsquared:.3f}",
            transform=ax.transAxes,ha="right",fontsize=8.5,color="#5f6368",style="italic",
            linespacing=1.5)
axes[0].set_ylabel("log-odds it is a real error")
fig.suptitle("Proper logistic regression on all raw rows (not binned summaries)\n"
             "Curvature is required in every dataset — but it bends UP for model-made errors "
             "and DOWN for hand-written distractors",
             fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.87])
fig.savefig(f"{FIG}/R4_regression_curves.png",dpi=150,facecolor="white")
print("wrote R4_regression_curves.png")
