import os
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results")+"/"
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B,O,A="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; SUB="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#9a9993","axes.labelcolor":INK,
  "xtick.color":SUB,"ytick.color":SUB,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
  "grid.color":GRID,"grid.linewidth":.6,"axes.axisbelow":True,"savefig.dpi":220,"savefig.bbox":"tight"})
# ---- Fig 1 pipeline
fig,ax=plt.subplots(figsize=(7.2,3.3)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,46)
def box(x,y,w,h,t,c="#f3f2ee",ec="#9a9993",fs=7.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=c,ec=ec,lw=1)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.35)
box(1,16,17,16,"Patient table\nDp140 group,\nMRI/DTI regions,\ncovariates")
box(23,16,17,16,"1. Data checks\nmissing values,\ncovariates")
box(45,16,19,16,"2. Group statistics\neffect size + CI,\nmax-T permutation",c="#dbe8f8",ec=B)
box(69,26,30,16,"3. Leakage-free pipeline\n(refit in every fold)\nimpute, scale, select k,\nregularised model",c="#dbe8f8",ec=B,fs=7)
box(69,3,30,16,"4. Trust checks\npermutation null,\nlearning curve,\npower, stability",c="#fbe3d8",ec=O,fs=7)
for x0,x1 in [(18.4,22.6),(40.4,44.6)]: ax.annotate("",xy=(x1,24),xytext=(x0,24),arrowprops=dict(arrowstyle="->",color=SUB))
ax.annotate("",xy=(68.6,30),xytext=(64.4,24),arrowprops=dict(arrowstyle="->",color=SUB))
ax.annotate("",xy=(84,19.5),xytext=(84,25.6),arrowprops=dict(arrowstyle="->",color=SUB))
ax.text(1,41,"Outer loop: repeated stratified 5-fold\nInner loop: 3-fold grid search over k and C",fontsize=7.5,color=SUB,va="center")
fig.savefig(RES+"figures/fig1_pipeline.png"); plt.close()
# ---- Fig 2 forest of published volumes
g=pd.read_csv(RES+"group_stats_doorenweerd2014.csv")
fig,ax=plt.subplots(figsize=(5.4,2.9)); g=g.sort_values("hedges_g"); y=np.arange(len(g))
ax.hlines(y,g.ci_low,g.ci_high,color=B,lw=2); ax.plot(g.hedges_g,y,"o",color=B,ms=7,mec="white",mew=1.5)
ax.axvline(0,color=SUB,lw=1); ax.set_yticks(y); ax.set_yticklabels(g.measure)
for yi,(gg,p) in enumerate(zip(g.hedges_g,g.p_holm)): ax.text(1.75,yi,f"g = {gg:.2f}, Holm p = {p:.2f}",va="center",fontsize=8,color=INK)
ax.set_xlim(-1,2.9); ax.set_xlabel("Hedges g, Dp140+ minus Dp140− (12 vs 12), 95% CI"); ax.grid(axis="y",visible=False)
fig.savefig(RES+"figures/fig2_forest.png"); plt.close()
# ---- Fig 3 leakage + power (simulation)
lc=pd.read_csv(RES+"simulation/learning_curve_power.csv")
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.9))
n0=lc[lc.d==0]; a1.plot(n0.n,n0.mean_leaky_auc,"-o",color=O,lw=2,ms=5,label="Selection before CV (leaky)"); a1.plot(n0.n,n0.mean_auc,"-o",color=B,lw=2,ms=5,label="In-fold pipeline")
a1.axhline(.5,color=SUB,lw=1,ls=":"); a1.set_ylim(.45,.78); a1.set_xlabel("Sample size n"); a1.set_ylabel("AUC on pure noise"); a1.legend(frameon=False,fontsize=7.5,loc="upper right"); a1.set_title("(a) Leakage inflates AUC on noise",fontsize=9,loc="left")
for d,c in [(0.5,B),(0.8,O),(1.2,A)]:
    s=lc[lc.d==d]; a2.plot(s.n,s.power*100,"-o",color=c,lw=2,ms=5,label=f"d = {d}")
a2.axhline(80,color=SUB,lw=1,ls=":"); a2.set_ylim(0,105); a2.set_xlabel("Sample size n"); a2.set_ylabel("Power (%)"); a2.legend(frameon=False,fontsize=7.5,loc="lower right"); a2.set_title("(b) Power of the classifier",fontsize=9,loc="left")
fig.tight_layout(); fig.savefig(RES+"figures/fig3_sim_leak_power.png"); plt.close()
# ---- Fig 4 simulated AUC at realistic effects
s3=pd.read_csv(RES+"nested_cv_scenarios.csv"); s4=pd.read_csv(RES+"trust_checks_summary.csv")
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.9))
for d,c in [(0.4,B),(0.6,O)]:
    s=s4[s4.d==d]; a1.plot(s.n,s.auc,"-o",color=c,lw=2,ms=5,label=f"d = {d}")
    a1.plot(s.n,s.null95,"--",color=c,lw=1,alpha=.7)
    a2.plot(s.n,s.perm_significant*100,"-o",color=c,lw=2,ms=5,label=f"d = {d}")
a1.axhline(.5,color=SUB,lw=1,ls=":"); a1.set_xlabel("Sample size n"); a1.set_ylabel("Mean cross-validated AUC"); a1.legend(frameon=False,fontsize=7.5); a1.set_title("(a) AUC (dashed: null 95th pct)",fontsize=9,loc="left")
a2.set_ylim(-3,105); a2.set_xlabel("Sample size n"); a2.set_ylabel("Datasets beating the null (%)"); a2.legend(frameon=False,fontsize=7.5); a2.set_title("(b) Permutation test, 10 datasets each",fontsize=9,loc="left")
fig.tight_layout(); fig.savefig(RES+"figures/fig4_sim_realistic.png"); plt.close()
# ---- Fig 5 ABIDE
gs=pd.read_csv(RES+"abide_group_statistics.csv"); rl=pd.read_csv(RES+"abide_learning_curve.csv")
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.9))
o=gs.assign(a=gs.cohen_d.abs()).sort_values("a",ascending=False)
ds=o.a.values; sig=(o.p_perm_FWER.values<.05)
a1.bar(range(len(ds)),ds,color=[O if s else B for s in sig],width=.8)
a1.set_xlabel("Cortical regions, ranked"); a1.set_ylabel("|Cohen's d|"); a1.set_title("(a) Autism vs control, 67 regions",fontsize=9,loc="left")
a1.text(len(ds)*.35,ds[0]*.9,"orange: survives family-wise\ncorrection (1 of 67)",fontsize=7.5,color=INK,va="top"); a1.grid(axis="x",visible=False)
a2.fill_between(rl.n,rl.min_auc,rl.max_auc,color=B,alpha=.15,lw=0); a2.plot(rl.n,rl.mean_auc,"-o",color=B,lw=2,ms=5,label="Mean AUC (range shaded)")
a2.plot(rl.n,rl.null95,"--",color=O,lw=1.5,label="Null 95th percentile")
for n,m,p in zip(rl.n,rl.mean_auc,rl.power): a2.text(n,.2,f"{int(p*100+0.5)}%",ha="center",fontsize=7.5,color=SUB)
a2.text(rl.n.mean(),.12,"Share of draws beating null",ha="center",fontsize=7.5,color=SUB)
a2.axhline(.5,color=SUB,lw=1,ls=":"); a2.set_ylim(.05,.95); a2.set_xlabel("Sample size n"); a2.set_ylabel("Cross-validated AUC"); a2.legend(frameon=False,fontsize=7.5,loc="upper left"); a2.set_title("(b) Learning curve, 40 draws per size",fontsize=9,loc="left")
fig.tight_layout(); fig.savefig(RES+"figures/fig5_abide.png"); plt.close()
print("ok")
