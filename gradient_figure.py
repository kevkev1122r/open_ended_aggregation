import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
G=json.load(open(f"{HERE}/results/control_gradient.json"))
fig,axes=plt.subplots(1,2,figsize=(12.6,4.9))
for ax,cfg in zip(axes,["qa","dialogue"]):
    rows=G["gradient"][cfg]; x=np.arange(len(rows))
    lab=[r["r"] if r["r"]!="random" else "rand" for r in rows]
    ax.plot(x,[r["r2"] for r in rows],"o-",color="#b3261e",lw=2,ms=7,label="R² of a STRAIGHT line")
    ax.plot(x,[r["r2_quad"] for r in rows],"s-",color="#0b8043",lw=2,ms=6,label="R² allowing curvature")
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0.2,1.03)
    ax.set_xlabel("controls = r-th nearest other question   (left = hardest)")
    ax.set_ylabel("goodness of fit (R²)"); ax.grid(alpha=0.25)
    a2=ax.twinx(); a2.plot(x,[r["slope"] for r in rows],"^--",color="#1a73e8",lw=1.5,ms=6,label="fitted β")
    a2.set_ylabel("fitted β", color="#1a73e8"); a2.tick_params(axis="y",colors="#1a73e8")
    ax.set_title(f"HaluEval {cfg}", fontsize=12, fontweight="bold")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=8.5,loc="lower right")
    ax.axvspan(-0.4,1.5,color="#b3261e",alpha=0.06)
    ax.text(0.05,0.26,"the regime the\naggregator lives in",fontsize=8.5,color="#b3261e",style="italic")
fig.suptitle("How much of R²=0.99 was topic-matching? Most of it.\n"
             "The relationship stays real and highly predictable — but it is CURVED, not straight",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.88])
fig.savefig(f"{HERE}/results/figures/R2_control_gradient.png",dpi=150,facecolor="white")
print("wrote R2_control_gradient.png")
