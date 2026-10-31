# ML_Project
Can we predict whether a Steam game will be well received (at least 80% positive reviews) using only information available at release, such as price, genre, platforms, supported languages and age rating?

This is a binary classification task. Games are labelled "well received" if at least 80% of their reviews are positive, using only games with 50 or more reviews so that the label is reliable. Review counts, owners, playtime and scores are excluded from the features because they are only known after release (data leakage).

## Dataset:
<p>-Name: Steam Games Dataset (Fronkon Games)<br>
-Source: https://huggingface.co/datasets/FronkonGames/steam-games-<br>
-Licence: CC by 4.0<br>
-Version Used: 20th June version, 125855 rows x 39 cols<br>
-Accessed 03/10/26<br>
Raw data is not included in repository. See setup below.</p>

## Setup:
1. Clone repository

```text
git clone https://github.com/knjokuuu/ML_Project.git
cd ML_Project
```

2. Create an environment and install dependencies:
with pip:
```text
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

3. Download the data:
Go to https://huggingface.co/datasets/FronkonGames/steam-games-dataset
Download `games.csv` (20th June version, 125,855 rows)
Save as data/games.csv
Note the csv header has a bug, so code supplies its own corrected column names. Do not change csv file

## Run the audit and baseline
From the repository root, run:

```text
python src/01_audit [path/to/games.csv]
python src/02_baseline.py [path/to/games.csv]
```

The audit reports the raw and modelling-subset shape, variable types, missing
values, duplicate rows and IDs, target distribution, and obvious outliers. It
saves the variable audit table and figures to `results/`.

The baseline script uses the same protocol for a majority-class dummy model,
logistic regression, and random forest. It makes an 80/20 stratified train/test
split, selects the model using stratified 5-fold cross-validation on the training
set only, and evaluates the selected random forest once on the held-out test set.
All imputation, scaling, and top-k genre/category vocabulary learning occur
inside the pipeline on each training fold.

The proposed metrics are ROC-AUC (primary), PR-AUC, F1, and accuracy. Results
are saved to `results/model_comparison.csv`, `results/cv_fold_scores.csv`, and
`results/random_forest_metrics.csv`; diagnostic figures are saved to
`results/figures/`.

## Notes:
- AI assistance is used in this project and will be declared in assistance.