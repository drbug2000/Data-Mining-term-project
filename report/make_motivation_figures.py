import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT="/home/gyuch_an02/marijuana/Data-Mining-term-project/report"

# Fig 1: smoothness assumption (query-query sim bin -> clicked-ad sim), r=0.336
labels=["[-1,0.2]","[0.2,0.4]","[0.4,0.6]","[0.6,0.8]","[0.8,1.0]"]
vals=[0.211,0.277,0.323,0.345,0.419]
ns=[7771,13408,6200,1997,612]
fig,ax=plt.subplots(figsize=(5.0,3.2))
x=np.arange(len(labels))
bars=ax.bar(x,vals,color="#3b6fb5",width=0.62)
# trend line
z=np.polyfit(x,vals,1);ax.plot(x,np.poly1d(z)(x),"--",color="#d2691e",lw=2,label="monotone trend (r=0.336)")
for i,(v,nn) in enumerate(zip(vals,ns)):
    ax.text(i,v+0.008,f"{v:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.text(i,0.012,f"n={nn}",ha="center",va="bottom",fontsize=7,color="white")
ax.set_xticks(x);ax.set_xticklabels(labels,fontsize=8)
ax.set_xlabel("query-query embedding cosine similarity bin",fontsize=9)
ax.set_ylabel("mean clicked-ad similarity\n(within pair)",fontsize=9)
ax.set_ylim(0,0.47);ax.legend(fontsize=8,loc="upper left")
ax.set_title("Smoothness assumption: similar queries click on similar ads",fontsize=9)
fig.tight_layout();fig.savefig(f"{OUT}/fig_smoothness.png",dpi=150);plt.close(fig)

# Fig 2: K-NN signal validity (SKNCP vs random), d=1.27
fig,ax=plt.subplots(figsize=(4.0,3.2))
b=ax.bar(["SKNCP\n(clicked ads of\nsimilar queries)","random ads\n(baseline)"],[0.884,0.677],
         color=["#2e7d32","#9e9e9e"],width=0.55)
for rect,v in zip(b,[0.884,0.677]):
    ax.text(rect.get_x()+rect.get_width()/2,v+0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.annotate("diff +0.208\nCohen's d = 1.27",xy=(0,0.884),xytext=(0.55,0.80),
            fontsize=8,color="#c62828",
            arrowprops=dict(arrowstyle="->",color="#c62828"))
ax.set_ylim(0,1.0);ax.set_ylabel("max cosine sim to current clicked ad",fontsize=9)
ax.set_title("K-NN signal validity",fontsize=9)
fig.tight_layout();fig.savefig(f"{OUT}/fig_knn_validity.png",dpi=150);plt.close(fig)
print("figures written:",OUT)
