import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# 0) Change working directory to the script’s folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1) Load your data
df = pd.read_csv("merged_movie_data.csv")
reddit = pd.read_csv("reddit_with_sentiment.csv", quotechar='"', on_bad_lines="skip")

# 2) Compute mean/std sentiment per movie
sent_stats = (
    reddit.groupby("movie_title")["comment_sentiment"]
    .agg(mean_sent="mean", std_sent="std")
    .reset_index()
)
sent_stats[["mean_sent", "std_sent"]] = sent_stats[["mean_sent", "std_sent"]].fillna(
    0.0
)

df = df.merge(sent_stats, left_on="Title", right_on="movie_title", how="left")
df[["mean_sent", "std_sent"]] = df[["mean_sent", "std_sent"]].fillna(0.0)

# 3) Engineer interaction features
df["weighted_sent"] = df["mean_sent"] * np.log1p(df["imdbVotes"])
df["sent_diff"] = df["mean_sent"] - (df["imdbRating"] / 10.0)

# 4) Prepare target and clean categorical columns
df["Log_BoxOffice"] = np.log1p(df["BoxOffice"])
df["Rated"] = df["Rated"].str.upper().str.strip()
df["Genre"] = df["Genre"].str.split(",").str[0].str.lower()

# 5) Define feature lists
meta_feats = [
    "Year",
    "Runtime",
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Rotten Tomatoes",
    "Metacritic",
    "weighted_sent",
    "sent_diff",
]
cat_feats = ["Rated", "Genre"]
FEATURES = meta_feats + cat_feats

X = df[FEATURES]
y = df["Log_BoxOffice"]

# 6) Build preprocessing + model pipeline
preprocessor = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="mean"), meta_feats),
        (
            "cat",
            Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ]
            ),
            cat_feats,
        ),
    ]
)

pipe = Pipeline(
    [("prep", preprocessor), ("rf", RandomForestRegressor(random_state=42))]
)

# 7) Evaluate with 5-fold cross-validation
cv = KFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)

print(
    "With engineered sentiment interactions:",
    np.round(scores, 3),
    "mean",
    scores.mean().round(3),
)
