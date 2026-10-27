import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data")
os.makedirs(RES, exist_ok=True)
import json, numpy as np, pandas as pd
import simulate_pipeline as sp
from joblib import Parallel, delayed
d=pd.read_csv(os.path.join(DATA,"abide_clean.csv"))
d=d[(d.AGE_AT_SCAN>=7)&(d.AGE_AT_SCAN<=18)].reset_index(drop=True)
feat=[c for c in d.columns if c.startswith("roi")]
# site harmonisation: z-score each feature within site (uses no labels)
g=d.groupby("SITE_ID")[feat]
d[feat]=(d[feat]-g.transform("mean"))/g.transform("std").replace(0,np.nan)
d[feat]=d[feat].fillna(0)
y=(d.DX_GROUP==1).astype(int).values            # 1 = autism, 0 = control
d["age"]=d.AGE_AT_SCAN
sp.NAMES=feat
rng=np.random.default_rng(2026)
print("n",len(d),"autism",y.sum(),"control",(1-y).sum(),"features",len(feat))
# 1. group statistics
gs=sp.group_statistics(d,y,rng,n_perm=5000)
gs.round(4).to_csv(os.path.join(RES,"abide_group_statistics.csv"),index=False)
top=gs.reindex(gs.cohen_d.abs().sort_values(ascending=False).index).head(8)
print("median |d|",round(gs.cohen_d.abs().median(),3),"max |d|",round(gs.cohen_d.abs().max(),3),
      "| features with FWER p<0.05:",int((gs.p_perm_FWER<0.05).sum()),"| raw p<0.05:",int((gs.p_perm<0.05).sum()))
print(top[["feature","cohen_d","d_ci_low","d_ci_high","d_age_adjusted","p_perm","p_perm_FWER"]].round(3).to_string(index=False))
X=d[feat].values
# 2. nested CV on all data
cv,sel=sp.nested_cv(X,y,seed=1,n_repeats=3)
print("nested CV mean",cv[["auc","balanced_acc","f1"]].mean().round(3).to_dict(),"fold SD auc",round(cv.auc.std(),3))
# 3. permutation test on all data
obs,null,p=sp.permutation_null(X,y,seed=1,n_perm=200)
print("OOF AUC",round(obs,3),"null mean",round(null.mean(),3),"null95",round(np.percentile(null,95),3),"p",round(p,4))
# 4. learning curve: stratified subsamples of size n, 40 draws each; null at same n
def draw(n,seed,shuffle=False):
    r=np.random.default_rng(seed)
    i1=r.choice(np.where(y==1)[0],n//2,replace=False); i0=r.choice(np.where(y==0)[0],n-n//2,replace=False)
    idx=np.r_[i1,i0]; yy=y[idx].copy()
    if shuffle: yy=r.permutation(yy)
    return sp.oof_auc(X[idx],yy,seed)
rows=[]
for n in (24,40,60,100,150):
    real=Parallel(n_jobs=-1)(delayed(draw)(n,s) for s in range(40))
    nl=Parallel(n_jobs=-1)(delayed(draw)(n,1000+s,True) for s in range(80))
    thr=np.percentile(nl,95)
    rows.append(dict(n=n,mean_auc=np.mean(real),sd_auc=np.std(real),min_auc=np.min(real),max_auc=np.max(real),null95=thr,power=float(np.mean(np.array(real)>thr))))
    print(rows[-1])
lc=pd.DataFrame(rows).round(3); lc.to_csv(os.path.join(RES,"abide_learning_curve.csv"),index=False)
# 5. stability
S=sel>=0.7
print("features selected in >=70% of outer folds:",int(S.sum()),"| top-5 selection frequency:",np.sort(sel)[-5:].round(2))
json.dump(dict(n=len(d),autism=int(y.sum()),control=int((1-y).sum()),nested=cv[["auc","balanced_acc","f1"]].mean().round(3).to_dict(),
  oof_auc=round(obs,3),null95=round(float(np.percentile(null,95)),3),perm_p=round(float(p),4),stable_features=int(S.sum())),open(os.path.join(RES,"abide_results.json"),"w"))
