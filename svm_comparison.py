"""Linear SVM versus logistic regression on the real ABIDE thickness data (same leakage-free pipeline)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data"); os.makedirs(RES, exist_ok=True)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import simulate_pipeline as sp

d = pd.read_csv(os.path.join(DATA, "abide_clean.csv"))
d = d[(d.AGE_AT_SCAN >= 7) & (d.AGE_AT_SCAN <= 18)].reset_index(drop=True)
feat = [c for c in d.columns if c.startswith("roi")]
g = d.groupby("SITE_ID")[feat]
d[feat] = ((d[feat] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0)   # same site z-scoring as abide_run.py
y = (d.DX_GROUP == 1).astype(int).values
X = d[feat].values

rows = []
for model in ("logreg", "svm"):
    cv, _ = sp.nested_cv(X, y, seed=1, n_repeats=3, model=model)
    rows.append(dict(analysis="nested CV, all 218", model=model, n=len(y), auc=cv.auc.mean(),
                     auc_sd=cv.auc.std(), balanced_acc=cv.balanced_acc.mean(), f1=cv.f1.mean()))

def draw(n, seed, model):
    r = np.random.default_rng(seed)
    i1 = r.choice(np.where(y == 1)[0], n // 2, replace=False); i0 = r.choice(np.where(y == 0)[0], n - n // 2, replace=False)
    idx = np.r_[i1, i0]
    return sp.oof_auc(X[idx], y[idx], seed, model=model)

for n in (24, 60, 100):
    for model in ("logreg", "svm"):
        a = np.array(Parallel(n_jobs=-1)(delayed(draw)(n, s, model) for s in range(40)))
        rows.append(dict(analysis=f"random subsamples, 40 draws", model=model, n=n, auc=a.mean(),
                         auc_sd=a.std(), balanced_acc=np.nan, f1=np.nan))
out = pd.DataFrame(rows).round(3)
out.to_csv(os.path.join(RES, "abide_svm_comparison.csv"), index=False)
print(out.to_string(index=False))
