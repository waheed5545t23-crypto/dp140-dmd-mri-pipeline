#!/usr/bin/env python3
"""
NOVA project - simulation of the Dp140 MRI/DTI pipeline on SYNTHETIC data.

Purpose: check that the planned pipeline is leakage-free and show how much to trust
a small-n classifier BEFORE real data is available. Nothing here is a real result.

Order of analysis (as recommended in the NOVA feedback):
  1. Group statistics first: Cohen's d + bootstrap CI, max-T permutation test (FWER),
     age-adjusted d.
  2. Cautious classifier second: regularised logistic regression inside ONE sklearn
     Pipeline (impute -> scale -> select -> model), nested CV, several metrics,
     feature-selection stability across folds.
  3. How far to trust it: permutation null, learning curves / power over n and effect
     size, and a demonstration of what data leakage does to the AUC.

Run:  python3 simulate_pipeline.py --outdir results/simulation            (about 3-6 minutes)
      python3 simulate_pipeline.py --outdir results/simulation --quick    (smoke test)

To use real data later: replace simulate() with a loader that returns
(DataFrame of region-level measures, y) with y = 1 for Dp140-, 0 for Dp140+.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 1. Synthetic data
# --------------------------------------------------------------------------- #
FEATURES = [
    ("FA_corpus_callosum", "FA"), ("FA_SLF", "FA"), ("FA_fornix", "FA"),
    ("FA_cerebellar_WM", "FA"),
    ("MD_corpus_callosum", "MD"), ("MD_SLF", "MD"), ("MD_cerebellar_WM", "MD"),
    ("MD_occipital_WM", "MD"),
    ("AD_cerebellar_WM", "AD"), ("AD_corpus_callosum", "AD"),
    ("thick_frontal", "CT"), ("thick_parietal", "CT"), ("thick_temporal", "CT"),
    ("vol_gray_matter", "VOL"), ("vol_cerebellum", "VOL"), ("vol_total_brain", "VOL"),
]
NAMES = [n for n, _ in FEATURES]
# Arbitrary placement of the "true" signal (direction loosely follows published
# Dp140- vs Dp140+ DTI findings: higher diffusivity, lower FA). Weight x d = shift in SD.
SIGNAL = {"MD_cerebellar_WM": 1.0, "AD_cerebellar_WM": 1.0,
          "MD_occipital_WM": 0.8, "FA_corpus_callosum": -0.8}
POS_FRACTION = 21 / 60  # Dp140+ share, as in the AJNR 2020 cohort (21 Dp140+, 39 Dp140-)


def simulate(n_total, d, rng, miss=0.05):
    """Return (X DataFrame incl. 'age', y). y=1 is Dp140-, y=0 is Dp140+."""
    n_pos = int(round(POS_FRACTION * n_total))
    n_neg = n_total - n_pos
    y = np.r_[np.zeros(n_pos, int), np.ones(n_neg, int)]
    fam = np.array([f for _, f in FEATURES])
    corr = np.where(fam[:, None] == fam[None, :], 0.35, 0.10)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n_total, len(FEATURES))) @ L.T
    shift = np.zeros(len(FEATURES))
    for name, w in SIGNAL.items():
        shift[NAMES.index(name)] = d * w
    X = Z + np.outer(y, shift)
    age = rng.normal(8.0, 1.2, n_total)          # matched between groups
    for j, (_, f) in enumerate(FEATURES):         # volumes depend on age (nuisance)
        if f == "VOL":
            X[:, j] += 0.3 * (age - 8.0) / 1.2
    X[rng.random(X.shape) < miss] = np.nan        # missing values (MCAR)
    df = pd.DataFrame(X, columns=NAMES)
    df["age"] = age
    return df, y


# --------------------------------------------------------------------------- #
# 2. Group statistics (run FIRST)
# --------------------------------------------------------------------------- #
def cohen_d(neg, pos):
    na, nb = len(pos), len(neg)
    sp = np.sqrt(((na - 1) * pos.var(ddof=1) + (nb - 1) * neg.var(ddof=1)) / (na + nb - 2))
    return (neg.mean() - pos.mean()) / sp


def boot_d(neg, pos, rng, B=2000):
    ia = rng.integers(0, len(pos), (B, len(pos)))
    ib = rng.integers(0, len(neg), (B, len(neg)))
    A, Bm = pos[ia], neg[ib]
    sp = np.sqrt(((len(pos) - 1) * A.var(axis=1, ddof=1) +
                  (len(neg) - 1) * Bm.var(axis=1, ddof=1)) / (len(pos) + len(neg) - 2))
    d = (Bm.mean(axis=1) - A.mean(axis=1)) / sp
    return np.percentile(d, [2.5, 97.5])


def adjusted_d(x, y, age):
    ok = ~np.isnan(x)
    M = np.column_stack([np.ones(ok.sum()), y[ok], age[ok]])
    beta, *_ = np.linalg.lstsq(M, x[ok], rcond=None)
    res = x[ok] - M @ beta
    return beta[1] / np.sqrt((res ** 2).sum() / (ok.sum() - 3))


def welch_t_many(x, P):
    """Welch t for many label vectors at once (rows of P); NaN-aware."""
    w = (~np.isnan(x)).astype(float)
    x0 = np.nan_to_num(x)
    Q = 1.0 - P
    n1, n0 = P @ w, Q @ w
    m1, m0 = (P @ x0) / n1, (Q @ x0) / n0
    v1 = ((P @ x0 ** 2) - n1 * m1 ** 2) / (n1 - 1)
    v0 = ((Q @ x0 ** 2) - n0 * m0 ** 2) / (n0 - 1)
    return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


def group_statistics(df, y, rng, n_perm=5000):
    P_obs = y[None, :].astype(float)
    P_perm = np.array([rng.permutation(y) for _ in range(n_perm)], dtype=float)
    t_obs, t_perm = [], []
    for f in NAMES:
        x = df[f].values
        t_obs.append(welch_t_many(x, P_obs)[0])
        t_perm.append(welch_t_many(x, P_perm))
    t_obs = np.array(t_obs)
    t_perm = np.array(t_perm)                       # (features, n_perm)
    max_t = np.abs(t_perm).max(axis=0)
    rows = []
    for j, f in enumerate(NAMES):
        x = df[f].values
        ok = ~np.isnan(x)
        neg, pos = x[ok & (y == 1)], x[ok & (y == 0)]
        lo, hi = boot_d(neg, pos, rng)
        rows.append({
            "feature": f, "n_pos": len(pos), "n_neg": len(neg),
            "cohen_d": cohen_d(neg, pos), "d_ci_low": lo, "d_ci_high": hi,
            "d_age_adjusted": adjusted_d(x, y, df["age"].values),
            "p_perm": (1 + (np.abs(t_perm[j]) >= abs(t_obs[j])).sum()) / (n_perm + 1),
            "p_perm_FWER": (1 + (max_t >= abs(t_obs[j])).sum()) / (n_perm + 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Classifier: ONE pipeline, every learned step is re-fitted inside each fold
# --------------------------------------------------------------------------- #
def make_pipe(k=5, C=0.1, model="logreg"):
    """model: 'logreg' (default, used for all main results) or 'svm' (linear SVM comparator)."""
    if model == "svm":
        clf = SVC(kernel="linear", C=C, class_weight="balanced")
    else:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", clf),
    ])


def nested_cv(X, y, seed, n_repeats=5, model="logreg"):
    """Outer: repeated stratified 5-fold. Inner: 3-fold grid search on k and C."""
    grid = {"select__k": [3, 5, 8], "clf__C": [0.01, 0.1, 1.0]}
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    rows, sel_counts, n_folds = [], np.zeros(X.shape[1]), 0
    for i, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(3, shuffle=True, random_state=seed + i)
        gs = GridSearchCV(make_pipe(model=model), grid, cv=inner, scoring="roc_auc", n_jobs=1)
        gs.fit(X[tr], y[tr])
        proba = gs.decision_function(X[te])   # monotone in the probability, so AUC is unchanged
        pred = gs.predict(X[te])
        rows.append({
            "auc": roc_auc_score(y[te], proba),
            "accuracy": accuracy_score(y[te], pred),
            "balanced_acc": balanced_accuracy_score(y[te], pred),
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "k": gs.best_params_["select__k"], "C": gs.best_params_["clf__C"],
        })
        sel_counts += gs.best_estimator_.named_steps["select"].get_support()
        n_folds += 1
    return pd.DataFrame(rows), sel_counts / n_folds


def oof_auc(X, y, seed, repeats=3, model="logreg"):
    """Pooled out-of-fold AUC (fixed hyper-parameters), averaged over CV repeats."""
    aucs = []
    for r in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        s = cross_val_predict(make_pipe(model=model), X, y, cv=cv, method="decision_function")
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs))


def leaky_oof_auc(X, y, seed, repeats=3, k=5):
    """WRONG on purpose: impute/scale/SELECT on the whole data, then cross-validate."""
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    Xk = SelectKBest(f_classif, k=k).fit(Xs, y).transform(Xs)
    aucs = []
    for r in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        s = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
                              Xk, y, cv=cv, method="decision_function")
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs))


def permutation_null(X, y, seed, n_perm=200):
    rng = np.random.default_rng(seed)
    obs = oof_auc(X, y, seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = Parallel(n_jobs=-1)(delayed(oof_auc)(X, yp, seed + 1 + i) for i, yp in enumerate(perms))
    null = np.array(null)
    return obs, null, (1 + (null >= obs).sum()) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# 4. How much to trust it: learning curve / power / leakage over n and effect size
# --------------------------------------------------------------------------- #
def one_dataset(n, d, seed, with_leaky):
    rng = np.random.default_rng(seed)
    df, y = simulate(n, d, rng)
    X = df[NAMES].values
    out = {"n": n, "d": d, "auc": oof_auc(X, y, seed)}
    if with_leaky:
        out["leaky_auc"] = leaky_oof_auc(X, y, seed)
    return out


def trust_study(sizes, effects, n_null, n_alt, seed0=10_000):
    jobs = []
    for n in sizes:
        for d in effects:
            reps = n_null if d == 0 else n_alt
            for r in range(reps):
                jobs.append((n, d, seed0 + 7919 * n + 131 * int(d * 100) + r, d == 0))
    res = Parallel(n_jobs=-1)(delayed(one_dataset)(*j) for j in jobs)
    return pd.DataFrame(res)


def summarise_trust(df):
    rows = []
    for n, g in df.groupby("n"):
        thr = np.percentile(g[g.d == 0]["auc"], 95)       # null AUC threshold at this n
        for d, gd in g.groupby("d"):
            rows.append({
                "n": n, "d": d, "mean_auc": gd.auc.mean(), "sd_auc": gd.auc.std(),
                "null_95pct_auc": thr,
                "power": float((gd.auc > thr).mean()) if d > 0 else 0.05,
                "mean_leaky_auc": gd.leaky_auc.mean() if "leaky_auc" in gd and d == 0 else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Figure (colour roles: validated default categorical slots 1-3 + neutral grey)
# --------------------------------------------------------------------------- #
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
C_BLUE, C_ORANGE, C_AQUA, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e89"


def style(ax, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def make_figure(gs_df, perm, sel_freq, summ, path, d_main):
    obs, null, p = perm
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURF)
    # A: forest plot of effect sizes
    ax = axs[0, 0]
    g = gs_df.sort_values("cohen_d").reset_index(drop=True)
    for i, r in g.iterrows():
        sig = r.p_perm_FWER < 0.05
        ax.plot([r.d_ci_low, r.d_ci_high], [i, i], color=C_BLUE if sig else C_GREY, lw=1.5)
        ax.plot(r.cohen_d, i, "o", ms=5.5, color=C_BLUE if sig else C_GREY,
                markeredgecolor=SURF, markeredgewidth=1.2)
        if r.feature in SIGNAL:
            ax.text(ax.get_xlim()[0], i, "", color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([("* " if f in SIGNAL else "  ") + f for f in g.feature], fontsize=7.5)
    ax.set_xlabel("Cohen's d (Dp140- minus Dp140+), 95% bootstrap CI")
    style(ax, "A. Group statistics first")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "blue = significant after FWER correction\n* = true signal (known only in simulation)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    # B: permutation null
    ax = axs[0, 1]
    ax.hist(null, bins=24, color=C_GREY, edgecolor=SURF, linewidth=1.2)
    ax.axvline(obs, color=C_BLUE, lw=2)
    ax.text(obs, ax.get_ylim()[1] * 0.92, f" observed AUC {obs:.2f}\n permutation p = {p:.3f}",
            color=INK, fontsize=8.5, va="top", ha="right" if obs > np.median(null) + 0.1 else "left")
    ax.set_xlabel("Cross-validated AUC under shuffled labels")
    ax.set_ylabel("Permutations")
    style(ax, "B. Is it above chance?")
    # C: feature stability
    ax = axs[0, 2]
    order = np.argsort(sel_freq)
    cols = [C_BLUE if NAMES[j] in SIGNAL else C_GREY for j in order]
    ax.barh(range(len(order)), sel_freq[order], color=cols, edgecolor=SURF, linewidth=1.2, height=0.75)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[j] for j in order], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of outer folds in which the feature was selected")
    style(ax, "C. Feature stability across folds")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    # D: learning curve
    colours = {0: C_GREY, 0.5: C_BLUE, 0.8: C_ORANGE, 1.2: C_AQUA}
    ax = axs[1, 0]
    for d, g in summ.groupby("d"):
        ax.plot(g.n, g.mean_auc, "-o", ms=5, lw=2, color=colours[d],
                markeredgecolor=SURF, markeredgewidth=1.2)
        ax.fill_between(g.n, g.mean_auc - g.sd_auc, g.mean_auc + g.sd_auc,
                        color=colours[d], alpha=0.14, lw=0)
        ax.text(g.n.iloc[-1] * 1.02, g.mean_auc.iloc[-1],
                "null (d=0)" if d == 0 else f"d={d}", color=INK, fontsize=8, va="center")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Total sample size n (35% Dp140+)")
    ax.set_ylabel("Mean CV AUC (band = 1 SD across datasets)")
    ax.set_xlim(right=summ.n.max() * 1.22)
    style(ax, "D. Learning curve: AUC vs n")
    # E: power
    ax = axs[1, 1]
    for d, g in summ[summ.d > 0].groupby("d"):
        ax.plot(g.n, g.power, "-o", ms=5, lw=2, color=colours[d],
                markeredgecolor=SURF, markeredgewidth=1.2)
        nudge = {0.5: -0.05, 0.8: -0.015, 1.2: 0.04}[d]   # keep end labels from colliding
        ax.text(g.n.iloc[-1] * 1.02, g.power.iloc[-1] + nudge, f"d={d}", color=INK, fontsize=8, va="center")
    ax.axhline(0.8, color=INK2, lw=0.8, ls=":")
    ax.text(summ.n.max() * 1.02, 0.8, "80%", color=INK2, fontsize=7.5, va="center")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(right=summ.n.max() * 1.22)
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Chance the classifier beats the null (5% level)")
    style(ax, "E. Power to detect the subgroup")
    # F: leakage demo
    ax = axs[1, 2]
    z = summ[summ.d == 0]
    ax.plot(z.n, z.mean_auc, "-o", ms=5, lw=2, color=C_BLUE, markeredgecolor=SURF, markeredgewidth=1.2)
    ax.plot(z.n, z.mean_leaky_auc, "-o", ms=5, lw=2, color=C_ORANGE, markeredgecolor=SURF, markeredgewidth=1.2)
    ax.text(z.n.iloc[-1] * 1.02, z.mean_auc.iloc[-1], "correct\n(in-fold)", color=INK, fontsize=8, va="center")
    ax.text(z.n.iloc[-1] * 1.02, z.mean_leaky_auc.iloc[-1], "leaky\n(select first)", color=INK, fontsize=8, va="center")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(right=z.n.max() * 1.3)
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Mean CV AUC on pure noise (d=0)")
    style(ax, "F. What leakage does (no real signal)")
    fig.suptitle(f"NOVA pipeline simulation on SYNTHETIC data (worked example: n=60, d={d_main}). Not real results.",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/simulation")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--d", type=float, default=0.8, help="effect size for the worked example")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- worked example: AJNR-like cohort (21 Dp140+, 39 Dp140-) ----
    df, y = simulate(60, a.d, rng)
    X = df[NAMES].values
    print(f"Worked example: n={len(y)}  Dp140+={int((y == 0).sum())}  Dp140-={int((y == 1).sum())}")

    print("1/4 group statistics ...")
    gs_df = group_statistics(df, y, rng, n_perm=1000 if a.quick else 5000)
    gs_df.round(4).to_csv(os.path.join(a.outdir, "group_statistics.csv"), index=False)

    print("2/4 nested CV ...")
    cv_df, sel_freq = nested_cv(X, y, a.seed, n_repeats=2 if a.quick else 5)
    cv_df.to_csv(os.path.join(a.outdir, "nested_cv_folds.csv"), index=False)

    print("3/4 permutation null ...")
    perm = permutation_null(X, y, a.seed, n_perm=40 if a.quick else 200)

    print("4/4 learning curve / power / leakage ...")
    sizes = [24, 40, 60] if a.quick else [24, 40, 60, 100, 160]
    effects = [0, 0.5, 0.8, 1.2]
    trust = trust_study(sizes, effects, n_null=20 if a.quick else 120, n_alt=15 if a.quick else 60)
    summ = summarise_trust(trust)
    summ.round(4).to_csv(os.path.join(a.outdir, "learning_curve_power.csv"), index=False)

    make_figure(gs_df, perm, sel_freq, summ, os.path.join(a.outdir, "nova_simulation.png"), a.d)

    metrics = cv_df[["auc", "accuracy", "balanced_acc", "precision", "recall", "f1"]]
    result = {
        "note": "SYNTHETIC data. Not real results.",
        "worked_example": {"n": int(len(y)), "d": a.d,
                           "n_Dp140_pos": int((y == 0).sum()), "n_Dp140_neg": int((y == 1).sum())},
        "nested_cv_mean": metrics.mean().round(3).to_dict(),
        "nested_cv_sd_across_folds": metrics.std().round(3).to_dict(),
        "nested_cv_best_k_counts": cv_df.k.value_counts().to_dict(),
        "nested_cv_best_C_counts": {str(k): int(v) for k, v in cv_df.C.value_counts().items()},
        "permutation": {"observed_oof_auc": round(perm[0], 3),
                        "null_mean": round(float(perm[1].mean()), 3),
                        "null_95pct": round(float(np.percentile(perm[1], 95)), 3),
                        "p_value": round(float(perm[2]), 4)},
        "feature_selection_frequency": {NAMES[j]: round(float(sel_freq[j]), 2) for j in np.argsort(-sel_freq)},
        "significant_features_FWER_0.05": gs_df[gs_df.p_perm_FWER < 0.05].feature.tolist(),
        "true_signal_features": list(SIGNAL),
    }
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("\nLearning curve / power table:")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
