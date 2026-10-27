import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data")
os.makedirs(RES, exist_ok=True)
import numpy as np, pandas as pd
from scipy import stats
# Doorenweerd 2014 (Ann Neurol) Table: mean, SD; Dp140+ n=12, Dp140- n=12 (T1w)
rows = {"Intracranial":(1633,149,1550,123),"Total brain":(1401,121,1324,114),
        "Grey matter":(770,53,749,63),"White matter":(631,75,576,61),"CSF":(232,39,226,29)}
n1=n2=12
out=[]
for k,(m1,s1,m2,s2) in rows.items():
    diff=m1-m2; sp=np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2)); d=diff/sp
    J=1-3/(4*(n1+n2)-9); g=d*J
    se=np.sqrt((n1+n2)/(n1*n2)+g**2/(2*(n1+n2)))
    lo,hi=g-1.96*se,g+1.96*se
    t,p=stats.ttest_ind_from_stats(m1,s1,n1,m2,s2,n2,equal_var=False)
    out.append(dict(measure=k,diff=diff,cohen_d=round(d,2),hedges_g=round(g,2),ci_low=round(lo,2),ci_high=round(hi,2),p_raw=p))
df=pd.DataFrame(out)
# Holm correction
o=df.p_raw.argsort().values; m=len(df); adj=np.empty(m); run=0
for r,i in enumerate(o):
    run=max(run,(m-r)*df.p_raw[i]); adj[i]=min(1,run)
df["p_holm"]=adj.round(3); df["p_raw"]=df.p_raw.round(3)
# power at n=12/12 for observed g, and n per group for 80% power
from scipy.optimize import brentq
def power(g,n,a=.05):
    df_=2*n-2; nc=g*np.sqrt(n/2); tc=stats.t.ppf(1-a/2,df_)
    return 1-stats.nct.cdf(tc,df_,nc)+stats.nct.cdf(-tc,df_,nc)
df["power_n12"]=[round(power(abs(g),12),2) for g in df.hedges_g]
df["n_per_group_80pct"]=[int(np.ceil(2*((1.96+0.8416)/abs(g))**2+1)) for g in df.hedges_g]
print(df.to_string(index=False)); df.to_csv(os.path.join(RES,"group_stats_doorenweerd2014.csv"),index=False)
