"""Preprocessing pipeline (scikit-learn). Everything learned from data (medians,
scaling, the top-k genre/category vocabulary) is fitted on the training data only,
because the whole thing is used inside an sklearn Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Right-skewed counts / prices: log(1 + x) then scale.
LOG_NUMERIC = ["Price", "Achievements", "n_languages", "n_audio_languages"]
PLAIN_NUMERIC = ["Required age", "release_year", "release_month"]
BINARY = ["Windows", "Mac", "Linux"]
LIST_COLUMNS = ["genres_list", "categories_list"]
FEATURE_COLUMNS = LOG_NUMERIC + PLAIN_NUMERIC + BINARY + LIST_COLUMNS


class MultiHotTopK(BaseEstimator, TransformerMixin):
    """Turn a column of lists into 0/1 columns for the k most common items.

    The vocabulary is learned in fit(), i.e. from the training data only.
    Items not in the vocabulary (including anything unseen in training) are ignored.
    """

    def __init__(self, k=20):
        self.k = k

    @staticmethod
    def _series(X):
        return X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X).ravel())

    def fit(self, X, y=None):
        s = self._series(X)
        self.prefix_ = X.columns[0].replace("_list", "") if hasattr(X, "columns") else "item"
        counts = pd.Series([v for lst in s for v in lst], dtype="object").value_counts()
        self.vocabulary_ = counts.head(self.k).index.tolist()
        return self

    def transform(self, X):
        s = self._series(X)
        index = {v: i for i, v in enumerate(self.vocabulary_)}
        out = np.zeros((len(s), len(self.vocabulary_)))
        for row, items in enumerate(s):
            for item in items:
                j = index.get(item)
                if j is not None:
                    out[row, j] = 1
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix_}_{v}" for v in self.vocabulary_])


def make_preprocessor(top_k=20):
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    plain_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("log_num", log_pipe, LOG_NUMERIC),
        ("num", plain_pipe, PLAIN_NUMERIC),
        ("bin", "passthrough", BINARY),
        ("genres", MultiHotTopK(top_k), ["genres_list"]),
        ("categories", MultiHotTopK(top_k), ["categories_list"]),
    ])