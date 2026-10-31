"""Data loading, label construction and release-time feature selection.

Used by the audit script and the baseline model script.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

# The raw CSV has a header bug: "DiscountDLC count" is really two columns.
COLS = ['AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU', 'Required age',
        'Price', 'Discount', 'DLC count', 'About the game', 'Supported languages',
        'Full audio languages', 'Reviews', 'Header image', 'Website', 'Support url',
        'Support email', 'Windows', 'Mac', 'Linux', 'Metacritic score', 'Metacritic url',
        'User score', 'Positive', 'Negative', 'Score rank', 'Achievements',
        'Recommendations', 'Notes', 'Average playtime forever',
        'Average playtime two weeks', 'Median playtime forever',
        'Median playtime two weeks', 'Developers', 'Publishers', 'Categories',
        'Genres', 'Tags', 'Screenshots', 'Movies']

MIN_REVIEWS = 50      # minimum Positive + Negative for a reliable label
CUTOFF = 0.80         # share of positive reviews counted as "well received"
RANDOM_STATE = 42

# Columns used to build the label or only known after release -> never features.
LEAKAGE = ['Positive', 'Negative', 'total', 'ratio', 'well_received', 'User score',
           'Metacritic score', 'Metacritic url', 'Recommendations', 'Estimated owners',
           'Peak CCU', 'Score rank', 'Reviews', 'Notes',
           'Average playtime forever', 'Average playtime two weeks',
           'Median playtime forever', 'Median playtime two weeks']

# Columns that are IDs, URLs, media or free text -> not used in this project.
NOT_USED = ['AppID', 'Name', 'Header image', 'Website', 'Support url', 'Support email',
            'Screenshots', 'Movies', 'About the game', 'Developers', 'Publishers']

# Usable columns that are snapshots of the scrape date rather than release time.
SNAPSHOT_RISK = ['Discount', 'DLC count', 'Tags']

# Raw columns the features are built from (Release date -> release_year / release_month).
FEATURE_SOURCES = ['Price', 'Required age', 'Achievements', 'Release date',
                   'Supported languages', 'Full audio languages',
                   'Windows', 'Mac', 'Linux', 'Genres', 'Categories']

# Kind of variable, for the data-audit table.
VARIABLE_TYPES = {
    'AppID': 'identifier', 'Name': 'identifier',
    'Release date': 'date (text)', 'Estimated owners': 'categorical range (text)',
    'Peak CCU': 'numeric (count)', 'Required age': 'numeric (ordinal)',
    'Price': 'numeric (continuous)', 'Discount': 'numeric (percent)', 'DLC count': 'numeric (count)',
    'About the game': 'free text', 'Supported languages': 'multi-valued text (list)',
    'Full audio languages': 'multi-valued text (list)', 'Reviews': 'free text',
    'Header image': 'url / media', 'Website': 'url / media', 'Support url': 'url / media',
    'Support email': 'url / media', 'Windows': 'boolean', 'Mac': 'boolean', 'Linux': 'boolean',
    'Metacritic score': 'numeric (score)', 'Metacritic url': 'url / media', 'User score': 'numeric (score)',
    'Positive': 'numeric (count)', 'Negative': 'numeric (count)', 'Score rank': 'numeric (rank)',
    'Achievements': 'numeric (count)', 'Recommendations': 'numeric (count)', 'Notes': 'free text',
    'Average playtime forever': 'numeric (minutes)', 'Average playtime two weeks': 'numeric (minutes)',
    'Median playtime forever': 'numeric (minutes)', 'Median playtime two weeks': 'numeric (minutes)',
    'Developers': 'multi-valued text (list)', 'Publishers': 'multi-valued text (list)',
    'Categories': 'multi-valued text (list)', 'Genres': 'multi-valued text (list)',
    'Tags': 'multi-valued text (list)', 'Screenshots': 'url / media', 'Movies': 'url / media',
}


def column_role(col):
    """Role of a raw column in this project."""
    if col in FEATURE_SOURCES:
        return 'feature'
    if col in ('Positive', 'Negative'):
        return 'builds the target (excluded from features)'
    if col in LEAKAGE:
        return 'excluded: leakage (known only after release)'
    if col in SNAPSHOT_RISK:
        return 'excluded: scrape-date snapshot'
    return 'excluded: identifier / text / media / unused'


NUMERIC = ['Price', 'Required age', 'Achievements', 'release_year', 'release_month',
           'n_languages', 'n_audio_languages']
BINARY = ['Windows', 'Mac', 'Linux']


def load_raw(path="data/games.csv"):
    return pd.read_csv(path, header=0, names=COLS)


def make_labelled_subset(df, min_reviews=MIN_REVIEWS, cutoff=CUTOFF):
    """Keep games with enough reviews and add the binary target."""
    df = df.copy()
    df["total"] = df["Positive"] + df["Negative"]
    sub = df[df["total"] >= min_reviews].copy()
    sub["ratio"] = sub["Positive"] / sub["total"]
    sub["well_received"] = (sub["ratio"] >= cutoff).astype(int)
    return sub.reset_index(drop=True)


def _parse_list(value):
    """Parse "['English', 'French']" style strings into a list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        out = ast.literal_eval(value)
        return list(out) if isinstance(out, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return [v.strip() for v in value.strip("[]").split(",") if v.strip()]


def _split_csv_field(value):
    """Split "Action,Indie" style strings into a list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def build_features(sub, top_k=20):
    """Return (X, y) using release-time information only.

    Row-wise transformations only (no statistics learned from other rows), so it is
    safe to call before the train/test split. Anything fitted on the data (imputing,
    scaling, choosing the top-k genres) should happen inside the sklearn Pipeline,
    on the training split only. The top_k categories here are therefore chosen
    by the caller from the TRAINING data, see `top_values`.
    """
    X = pd.DataFrame(index=sub.index)
    X["Price"] = pd.to_numeric(sub["Price"], errors="coerce")
    X["Required age"] = pd.to_numeric(sub["Required age"], errors="coerce")
    X["Achievements"] = pd.to_numeric(sub["Achievements"], errors="coerce")

    dates = pd.to_datetime(sub["Release date"], errors="coerce", format="mixed")
    X["release_year"] = dates.dt.year
    X["release_month"] = dates.dt.month

    langs = sub["Supported languages"].map(_parse_list)
    audio = sub["Full audio languages"].map(_parse_list)
    X["n_languages"] = langs.map(len)
    X["n_audio_languages"] = audio.map(len)

    for c in BINARY:
        X[c] = sub[c].astype(int)

    X["genres_list"] = sub["Genres"].map(_split_csv_field)
    X["categories_list"] = sub["Categories"].map(_split_csv_field)
    return X, sub["well_received"]


def top_values(series_of_lists, k=20):
    """Most frequent items in a column of lists (call on the training split only)."""
    return pd.Series([v for lst in series_of_lists for v in lst]).value_counts().head(k).index.tolist()


def multi_hot(series_of_lists, vocabulary, prefix):
    """One 0/1 column per vocabulary item."""
    return pd.DataFrame({f"{prefix}_{v}": series_of_lists.map(lambda lst, v=v: int(v in lst))
                         for v in vocabulary}, index=series_of_lists.index)