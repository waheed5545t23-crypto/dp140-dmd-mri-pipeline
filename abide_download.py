"""Download ABIDE ANTs regional cortical thickness and build ONE csv.

Run this on YOUR computer (it needs internet access to Amazon S3):
    pip install pandas requests
    python abide_download.py            # downloads about 300 subjects (default)
    python abide_download.py --n 1112   # all subjects (slower)

Output: abide_thickness_table.csv  (send this file back)
Columns: FILE_ID, SITE_ID, AGE_AT_SCAN, SEX, FIQ, DX_GROUP (1 = autism, 2 = control),
         followed by one column per cortical region (ANTs DKT thickness).

Sources (from the ABIDE Preprocessed documentation):
  phenotypic summary: .../ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
  thickness file:     .../ABIDE_Initiative/Outputs/ants/roi_thickness/<FILE_ID>_roi_thickness.txt
The parser is written defensively because the text layout is not documented; it prints
the first file so you can see what it contained.
"""
import argparse
import io
import re
import sys

import pandas as pd
import requests

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENO = f"{BASE}/Phenotypic_V1_0b_preprocessed1.csv"
THICK = f"{BASE}/Outputs/ants/roi_thickness/{{fid}}_roi_thickness.txt"


def parse(text):
    """Return {region: value}. Accepts 'name value' lines or 'name<TAB>value' lines,
    or a header row of names followed by a row of values."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # AFNI 3dROIstats layout: "File  Sub-brick  Mean_4  Mean_5 ..." then one data row
    # "<path> <sub-brick> v1 v2 ...". The path differs per subject, so use the header.
    if len(lines) >= 2 and lines[0].split()[0] == "File":
        names = lines[0].split()[2:]
        vals = lines[1].split()[-len(names):]
        if len(vals) == len(names):
            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
            return {n: num(v) for n, v in zip(names, vals)}
    out = {}
    for l in lines:
        parts = re.split(r"[\s,]+", l)
        if len(parts) >= 2:
            try:
                out[" ".join(parts[:-1])] = float(parts[-1])
            except ValueError:
                pass
    if out:
        return out
    if len(lines) >= 2:  # header row + values row
        names = re.split(r"[\s,]+", lines[0])
        vals = re.split(r"[\s,]+", lines[1])
        if len(names) == len(vals):
            return {n: float(v) for n, v in zip(names, vals)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of subjects to download")
    ap.add_argument("--out", default="abide_thickness_table.csv")
    a = ap.parse_args()

    ph = pd.read_csv(io.StringIO(requests.get(PHENO, timeout=60).text))
    ph = ph[ph["FILE_ID"] != "no_filename"].dropna(subset=["DX_GROUP"])
    # balanced draw across sites keeps both groups represented
    ph = ph.sample(frac=1, random_state=0).head(a.n)

    rows, shown = [], False
    for i, r in enumerate(ph.itertuples(index=False), 1):
        fid = r.FILE_ID
        resp = requests.get(THICK.format(fid=fid), timeout=60)
        if resp.status_code != 200:
            continue
        if not shown:
            print("First thickness file, first 10 lines:\n", "\n".join(resp.text.splitlines()[:10]))
            shown = True
        vals = parse(resp.text)
        if not vals:
            continue
        rows.append({"FILE_ID": fid, "SITE_ID": r.SITE_ID, "AGE_AT_SCAN": r.AGE_AT_SCAN,
                     "SEX": r.SEX, "FIQ": r.FIQ, "DX_GROUP": int(r.DX_GROUP), **vals})
        if i % 25 == 0:
            print(f"{i}/{len(ph)} downloaded", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"Saved {len(df)} subjects x {df.shape[1]} columns to {a.out}")


if __name__ == "__main__":
    main()
