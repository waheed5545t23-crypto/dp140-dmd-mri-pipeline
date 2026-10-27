import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data")
os.makedirs(RES, exist_ok=True)
import numpy as np, pandas as pd
import simulate_pipeline as sp
true_idx=[sp.NAMES.index(k) for k in sp.SIGNAL]
rows=[]; sel_all={}
for n in (24,60,100):
    for d in (0.4,0.6):
        sels=[]
        for rep in range(10):
            rng=np.random.default_rng(7000+n+int(d*100)*10+rep)
            df,y=sp.simulate(n,d,rng); X=df[sp.NAMES].values
            obs,null,p=sp.permutation_null(X,y,seed=rep,n_perm=100)
            _,sf=sp.nested_cv(X,y,seed=rep,n_repeats=2)
            sels.append(sf)
            noise=np.setdiff1d(np.arange(len(sf)),true_idx)
            top=set(np.argsort(sf)[-4:])
            rows.append(dict(n=n,d=d,rep=rep,oof_auc=obs,null95=np.percentile(null,95),p=p,
                             sig=p<0.05,sel_true=sf[true_idx].mean(),sel_noise=sf[noise].mean(),
                             top4_hits=len(top&set(true_idx))))
        S=np.array(sels)>=0.7
        # stability: mean pairwise Jaccard of the sets selected in >=70% of folds
        js=[]
        for i in range(10):
            for j in range(i+1,10):
                u=(S[i]|S[j]).sum(); js.append((S[i]&S[j]).sum()/u if u else np.nan)
        sel_all[(n,d)]=np.nanmean(js)
r=pd.DataFrame(rows); r.to_csv(os.path.join(RES,"trust_checks_runs.csv"),index=False)
s=r.groupby(["n","d"]).agg(perm_significant=("sig","mean"),auc=("oof_auc","mean"),null95=("null95","mean"),
        sel_true=("sel_true","mean"),sel_noise=("sel_noise","mean"),top4_hits=("top4_hits","mean")).round(2)
s["jaccard_stability"]=[round(sel_all[k],2) for k in s.index]
pd.set_option("display.width",200); print(s); s.to_csv(os.path.join(RES,"trust_checks_summary.csv"))
