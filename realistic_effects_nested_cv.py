import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data")
os.makedirs(RES, exist_ok=True)
import json, numpy as np, pandas as pd
import simulate_pipeline as sp
rows=[]
for n in (24,60):
    for d in (0.4,0.6):
        for rep in range(10):
            rng=np.random.default_rng(1000*n+int(d*100)*10+rep)
            df,y=sp.simulate(n,d,rng); X=df[sp.NAMES].values
            cv,_=sp.nested_cv(X,y,seed=rep,n_repeats=2)
            rows.append(dict(n=n,d=d,rep=rep,auc=cv.auc.mean(),bacc=cv.balanced_acc.mean(),f1=cv.f1.mean(),auc_sd_folds=cv.auc.std()))
r=pd.DataFrame(rows); r.to_csv(os.path.join(RES,"nested_cv_scenarios.csv"),index=False)
s=r.groupby(["n","d"]).agg(auc_mean=("auc","mean"),auc_sd_between_datasets=("auc","std"),auc_min=("auc","min"),auc_max=("auc","max"),bacc=("bacc","mean"),f1=("f1","mean"),fold_sd=("auc_sd_folds","mean")).round(3)
print(s); s.to_csv(os.path.join(RES,"nested_cv_summary.csv"))
