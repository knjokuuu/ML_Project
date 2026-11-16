# Steam game reception prediction

Can we predict whether a Steam game will be well received (at least 80% positive
reviews) using only information available at release, such as price, genre,
platforms, supported languages and age rating?

This is a binary classification project. A game is labelled **well received** if
at least 80% of its reviews are positive. Only games with at least 50 reviews
are included so that the label is reasonably reliable. Review counts, owners,
playtime and scores are excluded from the features because they are only known
after release and would cause data leakage.

## Dataset

- **Name:** Steam Games Dataset (Fronkon Games)
- **Source:** <https://huggingface.co/datasets/FronkonGames/steam-games->
- **Licence:** CC BY 4.0
- **Version used:** 20th June version, 125,855 rows x 39 columns
- **Accessed:** 03/10/26

The raw data is not included in the repository. See [Setup](#setup).

## Features and leakage control

The model uses 12 release-time features:

- price, required age and achievements;
- release year and month;
- number of supported and full-audio languages;
- Windows, macOS and Linux support;
- the 20 most common genres and categories, one-hot encoded.

Numeric count and price variables are log-transformed and scaled. Missing
numeric values are median-imputed. All imputation, scaling and vocabulary
learning happen inside the scikit-learn pipeline on each training fold.

Post-release variables (including review counts, owners, playtime, scores and
recommendations) are excluded. Scrape-date snapshot variables such as discount,
DLC count and tags are also excluded.

## Setup

1. Clone the repository:

   ```text
   git clone https://github.com/knjokuuu/ML_Project.git
   cd ML_Project
   ```

2. Create an environment and install dependencies:

   ```text
   python -m venv .venv
   ```

   Windows:

   ```text
   .venv\Scripts\activate
   ```

   macOS/Linux:

   ```text
   source .venv/bin/activate
   ```

   ```text
   pip install -r requirements.txt
   ```

3. Download `games.csv` from the dataset source. Use the 20th June version
   (125,855 rows) and save it as `data/games.csv`.

   The CSV header contains a known bug, so the project supplies corrected column
   names when loading it. Do not edit the downloaded file.

## Running the project

Run these commands from the repository root. Each command accepts an optional
path to a different copy of the dataset.

### 1. Data audit

```text
python src/01_audit.py [path/to/games.csv]
```

Reports dataset shape, variable types, missing values, duplicates, target
balance and numeric distributions. It writes the audit table and exploratory
figures to `results/`.

### 2. Baseline models

```text
python src/02_baseline.py [path/to/games.csv]
```

Compares a majority-class dummy model, logistic regression and default random
forest using an 80/20 stratified split and 5-fold cross-validation on the
training set. ROC-AUC is the primary metric; PR-AUC, F1 and accuracy are also
reported.

### 3. Model comparison and tuning

```text
python src/03_models.py [path/to/games.csv]
```

Compares the untuned baselines with default and tuned random forest and
histogram gradient-boosting models. Tuning uses nested cross-validation:
15 random-search configurations with 3 inner folds are selected separately
inside each of 5 outer folds. The held-out 20% test set remains closed and is
not used by this script.

Use `--quick` only to check that the script runs; quick results must not be
reported:

```text
python src/03_models.py [path/to/games.csv] --quick
```

The full run selects the tuned random forest as the best model. Its mean
5-fold training-set ROC-AUC is 0.7400 (SD 0.0090), with PR-AUC 0.7661,
F1 0.7306 and accuracy 0.6796.

### 4. Error analysis and robustness checks

Run this after the full model comparison:

```text
python src/04_error_analysis.py [path/to/games.csv]
```

This analyses the selected model using out-of-fold predictions, including:

- error rates by distance from the 80% cut-off, release year, review count,
  price and genre;
- the 15 most confident mistakes;
- permutation importance;
- feature ablation;
- stability across three random seeds.

The held-out test set remains closed for this analysis too.

## Results

Generated tables are stored in `results/`:

- `audit_variables.csv` — variable audit;
- `model_comparison.csv` and `cv_fold_scores.csv` — baseline results;
- `m3_model_comparison.csv` and `m3_cv_fold_scores.csv` — Milestone 3 model
  comparison and fold-level results;
- `best_params.json` — selected tuning settings;
- `m3_oof_predictions.csv` — out-of-fold predictions;
- `error_by_slice.csv` and `worst_errors.csv` — error analysis;
- `permutation_importance.csv` — feature importance;
- `ablation.csv` — feature ablation results;
- `seed_stability.csv` — random-seed stability results.

Figures are saved in `results/figures/`, including class balance, target rate by
year, numeric distributions, ROC/PR curves, calibration, error analysis and
permutation importance.

## Notes

AI assistance is used in this project and will be declared in the assistance
statement.
