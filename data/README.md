# Steam Games Dataset (FronkonGames)
<p>Source: https://huggingface.co/datasets/FronkonGames/steam-games-dataset<br>
Licence: CC BY 4.0<br>
Version: 20th June ver, 125,855 rows x 39 columns<br>
Accessed: 03/10/2026<br>
Collected from: Steam API and SteamSpy</p>

games.csv is not commited to this repository. Download games.csv from link above, and save as data/games.csv

The CSV header has a bug: the column "DiscountDLC count" is really two columns (Discount and DLC count), which shifts every later column label by one. src/data_prep.py supplies the corrected 40 column names when reading the file, so the CSV should be left unedited.