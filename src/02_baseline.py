"""Split design, preprocessing check and baselines under one protocol.

Models (same split, same 5 folds, same metrics):
  1. Majority-class dummy baseline
  2. Logistic regression
  3. Random forest

Model comparison uses cross-validation on the training set only. The 20% test set is
held out and NOT opened here; it is used once, for the final chosen model.

Run from the repo root:  python src/02_baseline.py [path/to/games.csv]
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import LEAKAGE, RANDOM_STATE, build_features, load_raw, make_labelled_subset
from pipeline import FEATURE_COLUMNS, make_preprocessor

path = sys.argv[1] if len(sys.argv) > 1 else "data/games.csv"
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

PRIMARY = "roc_auc"          # primary metric; the others are secondary / diagnostic
METRICS = ["roc_auc", "pr_auc", "f1", "accuracy"]

# ---------------------------------------------------------------- data
sub = make_labelled_subset(load_raw(path))
X, y = build_features(sub)
X = X[FEATURE_COLUMNS]
assert not set(X.columns) & set(LEAKAGE), "leakage column in features"
print(f"Modelling data: {X.shape[0]} games, {X.shape[1]} raw features, {y.mean():.3f} well received")

# ---------------------------------------------------------------- split design
# 80/20 stratified random split. Test set stays closed until the final model is chosen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {len(X_train)} ({y_train.mean():.3f} positive) | "
      f"Test (held out, not used here): {len(X_test)} ({y_test.mean():.3f} positive)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- preprocessing check
prep = make_preprocessor(top_k=20).fit(X_train)       # fitted on TRAIN only
Xt = prep.transform(X_train)
print(f"\nPreprocessed training matrix: {Xt.shape} ({len(prep.get_feature_names_out())} features)")
print("Genres kept     :", prep.named_transformers_["genres"].vocabulary_)
print("Categories kept :", prep.named_transformers_["categories"].vocabulary_)
print("NaNs after preprocessing:", int(pd.isna(Xt).sum()))

# ---------------------------------------------------------------- models
MODELS = {
    "Majority-class (dummy)": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
}

# The whole pipeline (imputer, scaler, genre vocabulary, model) is refit inside each fold.
fold_rows, oof_pred = [], {}
for name, estimator in MODELS.items():
    pipe = Pipeline([("prep", make_preprocessor(top_k=20)), ("clf", estimator)])
    oof = np.zeros(len(y_train), dtype=int)
    for k, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
        m = clone(pipe)
        t0 = time.perf_counter()
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = m.predict(X_train.iloc[va])
        proba = m.predict_proba(X_train.iloc[va])[:, 1]
        pred_s = time.perf_counter() - t0
        yv = y_train.iloc[va]
        oof[va] = pred
        fold_rows.append({
            "model": name, "fold": k,
            "roc_auc": roc_auc_score(yv, proba), "pr_auc": average_precision_score(yv, proba),
            "f1": f1_score(yv, pred), "accuracy": accuracy_score(yv, pred),
            "fit_seconds": fit_s, "predict_seconds": pred_s,
        })
    oof_pred[name] = oof
    print(f"finished {name}")

folds = pd.DataFrame(fold_rows)
folds.round(4).to_csv(RESULTS / "cv_fold_scores.csv", index=False)

summary = folds.groupby("model", sort=False).agg(
    **{f"{m}_mean": (m, "mean") for m in METRICS},
    **{f"{m}_sd": (m, "std") for m in METRICS},
    fit_seconds=("fit_seconds", "mean"), predict_seconds=("predict_seconds", "mean"))
summary.round(4).to_csv(RESULTS / "model_comparison.csv")

print(f"\n=== 5-fold CV on the training set (primary metric: {PRIMARY}) ===")
table = pd.DataFrame({m: summary[f"{m}_mean"].round(4).astype(str) + " +/- "
                      + summary[f"{m}_sd"].round(4).astype(str) for m in METRICS})
table["fit_s"] = summary["fit_seconds"].round(2)
print(table.to_string())

# ---------------------------------------------------------------- held-out test evaluation
# Fit the selected model once on all training data, then evaluate the untouched test set.
final_pipe = Pipeline([
    ("prep", make_preprocessor(top_k=20)),
    ("clf", clone(MODELS["Random forest"])),
])
final_pipe.fit(X_train, y_train)
test_pred = final_pipe.predict(X_test)
test_proba = final_pipe.predict_proba(X_test)[:, 1]
test_metrics = pd.DataFrame({
    "metric": METRICS,
    "value": [
        roc_auc_score(y_test, test_proba),
        average_precision_score(y_test, test_proba),
        f1_score(y_test, test_pred),
        accuracy_score(y_test, test_pred),
    ],
})
test_metrics.to_csv(RESULTS / "random_forest_metrics.csv", index=False)
print("\n=== Held-out test results (final random forest) ===")
print(test_metrics.round(4).to_string(index=False))

print("\nFold-level ROC-AUC (same folds for every model):")
print(folds.pivot(index="fold", columns="model", values="roc_auc")[list(MODELS)].round(4).to_string())

piv = folds.pivot(index="fold", columns="model", values=PRIMARY)
for a, b in [("Logistic regression", "Majority-class (dummy)"),
             ("Random forest", "Logistic regression")]:
    d = piv[a] - piv[b]
    print(f"{PRIMARY} difference {a} - {b}: mean {d.mean():+.4f}, sd {d.std():.4f}, "
          f"better in {(d > 0).sum()}/5 folds")

# ---------------------------------------------------------------- confusion matrices (out-of-fold)
print("\nOut-of-fold confusion matrices (rows = true 0/1, cols = predicted 0/1):")
fig, axes = plt.subplots(1, len(MODELS), figsize=(11.5, 4.4))
for ax, (name, oof) in zip(axes, oof_pred.items()):
    cm = confusion_matrix(y_train, oof)
    print(f"\n{name}\n{cm}")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() / 2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.sum() / 4 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Not well\nreceived", "Well\nreceived"])
    ax.set_yticks([0, 1], ["Not well\nreceived", "Well\nreceived"])
    ax.set_xlabel("Predicted")
    ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
    ax.grid(False)
axes[0].set_ylabel("Actual")
fig.suptitle("Confusion matrices from 5-fold cross-validation (training set only)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGURES / "04_confusion_matrices_cv.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved results/model_comparison.csv, results/cv_fold_scores.csv, "
      f"results/figures/04_confusion_matrices_cv.png")