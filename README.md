# ML_Project
Can we predict whether a Steam game will be well received (at least 80% positive reviews) using only information available at release, such as price, genre, platforms, supported languages and age rating?

This is a binary classification task. Games are labelled "well received" if at least 80% of their reviews are positive, using only games with 50 or more reviews so that the label is reliable. Review counts, owners, playtime and scores are excluded from the features because they are only known after release (data leakage).

Dataset:
    -Name: Steam Games Dataset (Fronkon Games)
    -Source: https://huggingface.co/datasets/FronkonGames/steam-games-dataset
    -Licence: CC by 4.0
    -Version Used: 20th June version, 125855 rows x 39 cols
    -Accessed 03/10/26
Raw data is not included in repository. See setup below.

Setup: 
    1. Clone repository -    git clone https://github.com/knjokuuu/ML_Project.git
    cd ML_Project

    2. Create environment and install dependencies - 
    with pip:
       python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt

    with conda: 
       conda env create -f environment.yml
   conda activate ml_project

    3. Download the data - 
        - Go to https://huggingface.co/datasets/FronkonGames/steam-games-dataset
        - Download games.csv (20th June ver, 115855 rows)
        - Save as data/games.csv
        Note the csv header has a bug, so code supplies its own corrected column names. Do not change csv file
Notes:
-AI assistance is used in this project and will be declared in assistance.