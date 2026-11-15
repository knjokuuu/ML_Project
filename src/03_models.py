"""Milestone 3: compare the baselines with two tuned learned models under ONE protocol.

Baselines (unchanged from Milestone 2, not tuned): majority-class dummy, logistic regression.
Learned models: random forest and histogram gradient boosting, each evaluated with default
settings and with settings tuned by random search.

Protocol
  * same stratified 80/20 split and the same 5 outer folds as Milestone 2;
  * tuned models use NESTED cross-validation: inside each outer training portion a random
    search (3 inner folds, ROC-AUC) picks the settings, the outer validation fold scores
    them, so the reported number is not inflated by the search;
  * the test set is never used here.

Run from the repo root:  python src/03_models.py [path/to/games.csv] [--quick]
(--quick shrinks the search so you can check the script runs; do not report quick results.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import LEAKAGE, RANDOM_STATE, build_features, load_raw, make_labelled_subset
from models import FIXED_MODELS, TUNED_MODELS, slug, with_prep
from pipeline import FEATURE_COLUMNS
from style import AQUA, BLUE, INK, MUTED, ORANGE, apply_style

parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default="data/games.csv")
parser.add_argument("--quick", action="store_true", help="tiny search, for checking only")
args = parser.parse_args()

N_ITER = 3 if args.quick else 15          # random-search configurations per search
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
apply_style()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

METRICS = ["roc_auc", "pr_auc", "f1", "accuracy"]
PRIMARY = "roc_auc"

# ---------------------------------------------------------------- data and split (as Milestone 2)
sub = make_labelled_subset(load_raw(args.data))
X, y = build_features(sub)
X = X[FEATURE_COLUMNS]
assert not set(X.columns) & set(LEAKAGE), "leakage column in features"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
print(f"Train: {len(X_train)} games ({y_train.mean():.3f} well received). "
      f"Test set: {len(X_test)} games, NOT used in this script.")
print(f"Random search: {N_ITER} configurations per search"
      f"{'  [QUICK MODE - not for reporting]' if args.quick else ''}\n")

# ---------------------------------------------------------------- evaluation loop
fold_rows, oof_proba = [], {}
all_names = list(FIXED_MODELS) + list(TUNED_MODELS)
for name in all_names:
    oof = np.zeros(len(y_train))
    for k, (tr, va) in enumerate(outer.split(X_train, y_train), start=1):
        Xtr, ytr, Xva, yva = X_train.iloc[tr], y_train.iloc[tr], X_train.iloc[va], y_train.iloc[va]
        search_s, best = 0.0, ""
        if name in FIXED_MODELS:
            model = with_prep(FIXED_MODELS[name]())
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
        else:
            factory, space = TUNED_MODELS[name]
            search = RandomizedSearchCV(with_prep(factory()), space, n_iter=N_ITER,
                                        scoring="roc_auc", cv=inner, n_jobs=1,
                                        random_state=RANDOM_STATE, refit=True)
            t0 = time.perf_counter()
            search.fit(Xtr, ytr)
            search_s = time.perf_counter() - t0
            fit_s = search.refit_time_
            model = search.best_estimator_
            best = json.dumps({k_.replace("clf__", ""): v for k_, v in search.best_params_.items()},
                              default=str)
        train_auc = roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1])   # overfitting check
        t0 = time.perf_counter()
        proba = model.predict_proba(Xva)[:, 1]
        pred = model.predict(Xva)
        pred_s = time.perf_counter() - t0
        oof[va] = proba
        fold_rows.append({
            "model": name, "fold": k, "train_roc_auc": train_auc,
            "roc_auc": roc_auc_score(yva, proba), "pr_auc": average_precision_score(yva, proba),
            "f1": f1_score(yva, pred), "accuracy": accuracy_score(yva, pred),
            "fit_seconds": fit_s, "predict_seconds": pred_s, "search_seconds": search_s,
            "chosen_settings": best,
        })
        print(f"  {name:<28s} fold {k}/5  ROC-AUC {fold_rows[-1]['roc_auc']:.4f}"
              + (f"  (search {search_s:.0f}s)" if search_s else ""), flush=True)
    oof_proba[name] = oof

folds = pd.DataFrame(fold_rows)
folds.to_csv(RESULTS / "m3_cv_fold_scores.csv", index=False)

summary = folds.groupby("model", sort=False).agg(
    **{f"{m}_mean": (m, "mean") for m in METRICS},
    **{f"{m}_sd": (m, "std") for m in METRICS},
    train_roc_auc_mean=("train_roc_auc", "mean"),
    fit_seconds=("fit_seconds", "mean"), predict_seconds=("predict_seconds", "mean"),
    search_seconds=("search_seconds", "mean"))
summary.round(4).to_csv(RESULTS / "m3_model_comparison.csv")

print(f"\n=== Results table: 5-fold CV on the training set (primary metric: {PRIMARY}) ===")
table = pd.DataFrame({m: summary[f"{m}_mean"].map("{:.4f}".format) + " +/- "
                      + summary[f"{m}_sd"].map("{:.4f}".format) for m in METRICS})
table["fit_s"] = summary["fit_seconds"].round(2)
table["predict_s"] = summary["predict_seconds"].round(3)
table["search_s"] = summary["search_seconds"].round(0)
print(table.to_string())
print("\n(tuned rows are nested-CV estimates; search_s = mean time of the inner random search "
      "per outer fold)")

print("\nOverfitting check (training vs validation ROC-AUC; a large gap means the model "
      "memorises the training data):")
gap = pd.DataFrame({"train_roc_auc": summary["train_roc_auc_mean"],
                    "validation_roc_auc": summary["roc_auc_mean"]})
gap["gap"] = gap["train_roc_auc"] - gap["validation_roc_auc"]
print(gap.round(4).to_string())

print("\nFold-level ROC-AUC (identical outer folds for every model):")
piv = folds.pivot(index="fold", columns="model", values=PRIMARY)[all_names]
print(piv.round(4).to_string())

print("\nPaired differences in ROC-AUC (A - B, per fold):")
pairs = [("Random forest (default)", "Logistic regression"),
         ("Gradient boosting (default)", "Logistic regression"),
         ("Random forest (tuned)", "Random forest (default)"),
         ("Gradient boosting (tuned)", "Gradient boosting (default)"),
         ("Random forest (tuned)", "Logistic regression"),
         ("Gradient boosting (tuned)", "Logistic regression"),
         ("Gradient boosting (tuned)", "Random forest (tuned)")]
for a, b in pairs:
    d = piv[a] - piv[b]
    print(f"  {a} - {b}: mean {d.mean():+.4f}, sd {d.std():.4f}, better in {(d > 0).sum()}/5 folds")

print("\nSettings chosen in each outer fold (shows how stable the tuning is):")
for name in TUNED_MODELS:
    print(f"\n{name}")
    for _, r in folds[folds["model"] == name].iterrows():
        print(f"  fold {int(r['fold'])}: {r['chosen_settings']}")

# ---------------------------------------------------------------- final settings (for the report)
# One more search on ALL training data with the same 5 folds. Its best score is optimistic
# (it is the winner of a search), so only the settings are reported, not the score.
final = {"n_iter": N_ITER, "quick_mode": bool(args.quick), "models": {}}
best_name = summary.loc[list(TUNED_MODELS), f"{PRIMARY}_mean"].idxmax()
final["best_model"] = best_name
for name, (factory, space) in TUNED_MODELS.items():
    print(f"\nFinal search for {name} on the full training set ...", flush=True)
    search = RandomizedSearchCV(with_prep(factory()), space, n_iter=N_ITER, scoring="roc_auc",
                                cv=outer, n_jobs=1, random_state=RANDOM_STATE, refit=False)
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)
    final["models"][name] = {
        "best_params": {k: v for k, v in search.best_params_.items()},
        "search_cv_roc_auc_optimistic": float(search.best_score_),
        "top5": [{"rank": int(r["rank_test_score"]), "roc_auc": float(r["mean_test_score"]),
                  "params": r["params"]} for _, r in res.iterrows()],
        "search_space": {k: [str(v) for v in vs] for k, vs in space.items()},
    }
    print("  best settings:", search.best_params_)
    print(f"  top configurations: {[round(float(s), 4) for s in res['mean_test_score']]}")
with open(RESULTS / "best_params.json", "w") as f:
    json.dump(final, f, indent=2, default=str)
print(f"\nCurrent best model by nested-CV {PRIMARY}: {best_name}")

# ---------------------------------------------------------------- out-of-fold predictions
oof_df = pd.DataFrame({"y_true": y_train.values,
                       "ratio": sub.loc[X_train.index, "ratio"].values,
                       "total": sub.loc[X_train.index, "total"].values},
                      index=X_train.index)
for name, p in oof_proba.items():
    oof_df[slug(name)] = p
oof_df.to_csv(RESULTS / "m3_oof_predictions.csv")

# ---------------------------------------------------------------- figures
SHOWN = {"Logistic regression": BLUE, "Random forest (tuned)": ORANGE,
         "Gradient boosting (tuned)": AQUA}
yt = y_train.values
prev = yt.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
for name, col in SHOWN.items():
    p = oof_proba[name]
    fpr, tpr, _ = roc_curve(yt, p)
    ax1.plot(fpr, tpr, color=col, linewidth=2, label=f"{name} (AUC {roc_auc_score(yt, p):.3f})")
    pr, rc, _ = precision_recall_curve(yt, p)
    ax2.plot(rc[:-1], pr[:-1], color=col, linewidth=2,
             label=f"{name} (AP {average_precision_score(yt, p):.3f})")
ax1.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Dummy / guessing (AUC 0.500)")
ax2.axhline(prev, color=MUTED, linestyle="--", linewidth=1.2, label=f"Dummy (AP {prev:.3f})")
ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
ax1.set_title("ROC curves")
ax2.set_xlabel("Recall (well received)"); ax2.set_ylabel("Precision")
ax2.set_ylim(0.4, 1.0)
ax2.set_title("Precision-recall curves")
ax1.legend(loc="lower right", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
fig.suptitle("Out-of-fold predictions pooled over the 5 outer folds (training set only)",
             x=0.01, ha="left", fontsize=10.5, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES / "05_roc_pr_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Perfectly calibrated")
for name, col in SHOWN.items():
    frac, mean_p = calibration_curve(yt, oof_proba[name], n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, color=col, linewidth=2, marker="o", markersize=6,
            markeredgecolor="#fcfcfb", label=name)
ax.set_xlabel("Mean predicted probability (well received)")
ax.set_ylabel("Observed share well received")
ax.set_title("Calibration (10 bins of equal size)")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "06_calibration.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nSaved: results/m3_model_comparison.csv, m3_cv_fold_scores.csv, best_params.json, "
      "m3_oof_predictions.csv, figures/05_roc_pr_curves.png, figures/06_calibration.png")