# Steam Games Dataset (FronkonGames)
Source: https://huggingface.co/datasets/FronkonGames/steam-games-dataset
Licence: CC BY 4.0
Version: 20th June ver, 125,855 rows x 39 columns
Accessed: 03/10/2026
Collected from: Steam API and SteamSpy

games.csv is not commited to this repository. Download games.csv from link above, and save as data/games.csv

The CSV header has a bug: the column "DiscountDLC count" is really two columns (Discount and DLC count), which shifts every later column label by one. src/data_prep.py supplies the corrected 40 column names when reading the file, so the CSV should be left unedited.