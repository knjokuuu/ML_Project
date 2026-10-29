import pandas as pd

cols = ['AppID','Name','Release date','Estimated owners','Peak CCU','Required age',
        'Price','Discount','DLC count','About the game','Supported languages',
        'Full audio languages','Reviews','Header image','Website','Support url',
        'Support email','Windows','Mac','Linux','Metacritic score','Metacritic url',
        'User score','Positive','Negative','Score rank','Achievements',
        'Recommendations','Notes','Average playtime forever',
        'Average playtime two weeks','Median playtime forever',
        'Median playtime two weeks','Developers','Publishers','Categories',
        'Genres','Tags','Screenshots','Movies']

df = pd.read_csv("data/games.csv", header=0, names=cols)
print(df.shape)
print(df["Windows"].isna().sum())
print(df[["Positive", "Negative"]].describe())

df["total"] = df["Positive"] + df["Negative"]
for n in [10, 20, 50, 100]:
    print(n, (df["total"] >= n).sum())

sub = df[df["total"] >= 50].copy()
sub["ratio"] = sub["Positive"] / sub["total"]
print(sub["ratio"].describe())
print((sub["ratio"] >= 0.8).mean())

print(df.isna().mean().sort_values(ascending=False).head(10))
print((sub["Price"] == 0).mean())

sub["well_received"] = (sub["ratio"] >= 0.8).astype(int)

print(sub.shape)
print(sub.dtypes)
print(sub.isna().mean().sort_values(ascending=False).head(15))
print(sub.duplicated(subset="AppID").sum())
print(sub["well_received"].value_counts(normalize=True))
print(sub[["Price", "Peak CCU", "total"]].describe())