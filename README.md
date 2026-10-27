# Dp140 and brain MRI/DTI in DMD: analysis code

Code behind the paper *Reliability of Brain MRI and DTI Patterns for Distinguishing Dp140 Subgroups in
Duchenne Muscular Dystrophy Using Leakage-Free Cross-Validated Classification* (Mahmoud Waheed,
IYNA - NOVA Research Program 2026, OCT Chapter).

**Important:** no DMD patient-level data are included or analysed. The scripts use (a) published group
means and SDs, (b) synthetic data, and (c) public ABIDE I cortical-thickness data (autism vs controls)
as a stand-in to test the pipeline. Synthetic results are not findings.

## Setup
Python 3.10+.

    pip install -r requirements.txt

Run every command from this folder.

## What to run (in this order)

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python simulate_pipeline.py` | Synthetic-data study: group statistics, nested CV, permutation null, learning curves, power, leakage demo (about 3 to 6 min; add `--quick` for a smoke test) | `results/simulation/` |
| 2 | `python published_group_stats.py` | Hedges g, CI, Holm p and power from the published Dp140+ vs Dp140- volumes (Doorenweerd 2014) | `results/group_stats_doorenweerd2014.csv` |
| 3 | `python realistic_effects_nested_cv.py` | Nested CV on 10 simulated datasets per setting at d = 0.4 and 0.6 (n = 24, 60) | `results/nested_cv_*.csv` |
| 4 | `python trust_checks.py` | Permutation test and feature stability on simulated data (n = 24, 60, 100) | `results/trust_checks_*.csv` |
| 5 | `python abide_prepare.py` | Cleans the ABIDE table in `data/` | `data/abide_clean.csv` |
| 6 | `python abide_run.py` | The same pipeline on real ABIDE thickness data (218 children aged 7 to 18) | `results/abide_*` |
| 7 | `python svm_comparison.py` | Linear SVM vs logistic regression on the ABIDE data, same leakage-free pipeline | `results/abide_svm_comparison.csv` |
| 8 | `python make_figures.py` | Draws the five paper figures from the result files | `results/figures/` |

Steps 3, 4 and 6 take several minutes each (they use all CPU cores).

## Getting the ABIDE table yourself
`data/abide_thickness_table.csv` is included. To rebuild it, run `python abide_download.py` (needs
internet access to Amazon S3; try Google Colab if your network blocks it), move the resulting CSV into
`data/`, then run step 5. Source: ABIDE Preprocessed (Craddock et al., 2013), ANTs regional cortical
thickness. Cite Di Martino et al., 2014 and Craddock et al., 2013 if you reuse it.

## Using your own Dp140 data
`simulate_pipeline.py` is the template. Replace `simulate()` with a loader that returns a DataFrame of
region-level features plus age, and a label vector `y` (1 = Dp140-, 0 = Dp140+). Everything downstream
(group statistics, nested CV, permutation test, learning curve) runs unchanged.

## Pipeline in one line
median imputation -> standardisation -> univariate feature selection (k = 3, 5, 8) -> logistic
regression with balanced class weights (a linear SVM is available as a comparator, `model="svm"`), all re-fitted inside every fold; outer repeated stratified
5-fold, inner 3-fold grid search over k and C.
