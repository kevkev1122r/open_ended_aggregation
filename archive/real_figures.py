import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG="results/figures"
GREY,RED,ORANGE,PURPLE,BLUE="#9aa0a6","#b3261e","#e8871a","#7a5195","#1a73e8"

M=["Random\n(floor)","Majority vote\n(exact match)","Cluster-then-vote\n(Universal Self-Consistency)",
   "Medoid\n(kernel, no weights)","KWA-EM\n(ours)"]
V=[62.21,83.03,82.81,82.88,86.25]
C=[GREY,RED,ORANGE,PURPLE,BLUE]

fig,ax=plt.subplots(figsize=(10.4,5.4))
b=ax.bar(M,V,color=C,width=.6)
for r,x in zip(b,V):
    ax.text(r.get_x()+r.get_width()/2,x+.9,f"{x:.2f}",ha="center",fontweight="bold",fontsize=11)
ax.axhline(V[1],ls=":",color=RED,lw=1.3)
ax.text(4.42,V[1]+.4,"majority-vote baseline",fontsize=8.5,color=RED,ha="right")
b[4].set_edgecolor("black"); b[4].set_linewidth(2.4)
ax.set_ylim(55,93); ax.set_ylabel("accuracy (%)")
ax.grid(axis="y",alpha=.25); ax.tick_params(axis="x",labelsize=9)
ax.set_title("Aggregating 6 models on 3,000 TriviaQA questions  (n = 2,757 scored)\n"
             'models prompted for the answer alone — e.g. "Tina Turner"',
             fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/X1_real_results.png",dpi=150,facecolor="white")
plt.close(fig); print("X1_real_results.png (single panel)")

fig,ax=plt.subplots(figsize=(9.6,4.8))
x=np.arange(2); w=.36
ker=[-0.15,7.29]; wei=[3.37,-0.76]
b1=ax.bar(x-w/2,ker,w,color=PURPLE,label="similarity kernel  (majority vote → medoid)")
b2=ax.bar(x+w/2,wei,w,color=BLUE,label="learned weights  (medoid → KWA)")
for bs,vals,ps in ((b1,ker,["p=0.73","p=8e-31"]),(b2,wei,["p=1e-11","p=0.13"])):
    for r,v,p in zip(bs,vals,ps):
        ax.text(r.get_x()+r.get_width()/2,v+(.35 if v>=0 else -.75),
                f"{v:+.2f}\n{p}",ha="center",fontsize=9,fontweight="bold")
ax.axhline(0,color="k",lw=1)
ax.set_xticks(x); ax.set_xticklabels(["short-answer prompt","full-sentence prompt"],fontsize=11)
ax.set_ylabel("accuracy points contributed"); ax.grid(axis="y",alpha=.25); ax.legend(fontsize=9.5)
ax.set_ylim(-2.6,9)
ax.set_title("The two halves of the method work in opposite settings — and never together",
             fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/X2_ingredients.png",dpi=150,facecolor="white")
print("X2_ingredients.png")
