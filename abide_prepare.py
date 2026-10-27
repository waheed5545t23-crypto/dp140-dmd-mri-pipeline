import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results"); DATA = os.path.join(HERE, "data")
os.makedirs(RES, exist_ok=True)
import pandas as pd, numpy as np, re
raw=pd.read_csv(os.path.join(DATA,"abide_thickness_table.csv"))
meta=["FILE_ID","SITE_ID","AGE_AT_SCAN","SEX","FIQ","DX_GROUP"]
if raw.shape[1] > 150:
    # table written by the FIRST version of abide_download.py: every subject's full vector
    # sits inside a column NAME (path + values), so recover it from there
    rows=[]
    for c in raw.columns[6:]:
        t=c.split(); fid=re.search(r"/([^/]+)_anat_thickness",t[0]).group(1)
        rows.append([fid]+[float(v) for v in t[2:]])
    nroi=len(rows[0])-1
    vec=pd.DataFrame(rows,columns=["FILE_ID"]+[f"roi{i+1:02d}" for i in range(nroi)])
    d=raw[meta].merge(vec,on="FILE_ID",how="inner")
else:
    # table written by the corrected abide_download.py: one column per region already
    d=raw.copy()
    d.columns=meta+[f"roi{i+1:02d}" for i in range(raw.shape[1]-len(meta))]
    nroi=raw.shape[1]-len(meta)
print("subjects",len(d),"| ROI values per subject",nroi)
d["FIQ"]=d.FIQ.where(d.FIQ>0)
roi=[c for c in d.columns if c.startswith("roi")]
keep=[c for c in roi if d[c].mean()>0.5]
print("cortical-like ROIs kept (mean thickness > 0.5 mm):",len(keep),"of",len(roi))
d=d[["FILE_ID","SITE_ID","AGE_AT_SCAN","SEX","FIQ","DX_GROUP"]+keep]
d.to_csv(os.path.join(DATA,"abide_clean.csv"),index=False)
p=d[(d.AGE_AT_SCAN>=7)&(d.AGE_AT_SCAN<=18)]
print("age 7-18:",len(p),p.DX_GROUP.map({1:"autism",2:"control"}).value_counts().to_dict(), "sites",p.SITE_ID.nunique())
print(d[keep].describe().T[["mean","std"]].describe().loc[["min","max"]])
