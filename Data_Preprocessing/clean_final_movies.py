import pandas as pd
import numpy as np
from datetime import datetime
import os
import csv

# 1. cd into script’s folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 2. Load the full dataset
df = pd.read_csv("final_movies.csv")


# 3. Normalize release dates into YYYY-MM-DD
def convert_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # Already ISO?
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return pd.to_datetime(s)
    # Try “DD Mon YYYY”
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            continue
    return pd.NaT


df["Released"] = df["Released"].apply(convert_date)

# 4. Strip units & cast numeric fields
df["Runtime"] = (
    df["Runtime"]
    .str.replace(" min", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)
df["Metascore"] = pd.to_numeric(df["Metascore"], errors="coerce")
df["imdbVotes"] = (
    df["imdbVotes"]
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)
df["BoxOffice"] = (
    df["BoxOffice"]
    .str.replace(r"[$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
df["Rotten Tomatoes"] = (
    df["Rotten Tomatoes"]
    .str.replace("%", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)
df["Metacritic"] = pd.to_numeric(df["Metacritic"], errors="coerce")

# Round the decimal ratings
df["imdbRating"] = df["imdbRating"].round(1)
df["Internet Movie Database"] = df["Internet Movie Database"].round(1)

# 5. Drop any rows where BoxOffice failed to parse at all
df = df[df["BoxOffice"].notna()]

# 6. Decide which numeric columns to drop vs. impute
numeric_cols = [
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Internet Movie Database",
    "Rotten Tomatoes",
    "Metacritic",
]
to_drop, to_impute = [], []
total = len(df)
for c in numeric_cols:
    miss = df[c].isna().sum()
    frac = miss / total
    if frac > 0.30:
        to_drop.append(c)
    elif miss > 0:
        to_impute.append(c)

df.drop(columns=to_drop, inplace=True)

for c in to_impute:
    df[c].fillna(df[c].mean(), inplace=True)

# 7. Write out one final CSV
df.to_csv("final_movies_cleaned.csv", index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Wrote {len(df)} rows to final_movies_cleaned.csv")
print(f"Dropped columns: {to_drop}")
print(f"Imputed columns: {to_impute}")
