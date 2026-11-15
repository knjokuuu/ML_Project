"""Model definitions and hyperparameter search spaces for Milestone 3.

Baselines (dummy, logistic regression) are exactly as in Milestone 2 and are not tuned.
The two learned models compared against them are a random forest (Week 7, bagging) and a
histogram gradient-boosting classifier (Week 7, boosting).
"""
import re

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_prep import RANDOM_STATE
from pipeline import make_preprocessor


def make_rf(**params):
    base = dict(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    return RandomForestClassifier(**{**base, **params})


def make_hgb(**params):
    # max_iter is a ceiling: early stopping on an internal 10% validation split ends
    # training when the score has not improved for 15 rounds (Week 7, early stopping).
    base = dict(early_stopping=True, n_iter_no_change=15, random_state=RANDOM_STATE)
    return HistGradientBoostingClassifier(**{**base, **params})


# Models evaluated exactly as in Milestone 2 (not tuned).
FIXED_MODELS = {
    "Majority-class (dummy)": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random forest (default)": lambda: make_rf(),
    "Gradient boosting (default)": lambda: make_hgb(),
}

RF_SPACE = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [None, 8, 12, 16, 24],
    "clf__min_samples_leaf": [1, 3, 5, 10, 20],
    "clf__max_features": ["sqrt", 0.3, 0.5],
}
HGB_SPACE = {
    "clf__learning_rate": [0.02, 0.05, 0.1, 0.2],
    "clf__max_leaf_nodes": [8, 15, 31, 63],
    "clf__min_samples_leaf": [20, 50, 100, 200],
    "clf__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "clf__max_iter": [200, 400, 600],
}

# name -> (estimator factory, search space)
TUNED_MODELS = {
    "Random forest (tuned)": (make_rf, RF_SPACE),
    "Gradient boosting (tuned)": (make_hgb, HGB_SPACE),
}

ESTIMATOR_FOR = {"Random forest (tuned)": make_rf, "Gradient boosting (tuned)": make_hgb}


def with_prep(estimator, exclude=()):
    """Full pipeline: preprocessing (fitted inside every fold) followed by the model."""
    return Pipeline([("prep", make_preprocessor(top_k=20, exclude=exclude)),
                     ("clf", estimator)])


def from_search_params(name, params, exclude=()):
    """Rebuild a tuned pipeline from a saved `best_params_` dict ('clf__x' keys)."""
    clean = {re.sub(r"^clf__", "", k): v for k, v in params.items()}
    return with_prep(ESTIMATOR_FOR[name](**clean), exclude=exclude)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")