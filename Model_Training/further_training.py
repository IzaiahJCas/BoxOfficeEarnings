import os
import pickle

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import shap

# 1) Data loading & sentiment feature engineering
os.chdir(os.path.dirname(os.path.abspath(__file__)))

movies = pd.read_csv("merged_movie_data.csv")
reddit = pd.read_csv("reddit_with_sentiment.csv", quotechar='"', on_bad_lines="skip")

sent_stats = (
    reddit.groupby("movie_title")["comment_sentiment"]
    .agg(mean_sent="mean", std_sent="std")
    .reset_index()
)
sent_stats[["mean_sent", "std_sent"]] = sent_stats[["mean_sent", "std_sent"]].fillna(
    0.0
)

df = movies.merge(sent_stats, left_on="Title", right_on="movie_title", how="left")
df[["mean_sent", "std_sent"]] = df[["mean_sent", "std_sent"]].fillna(0.0)
df["Log_BoxOffice"] = np.log1p(df["BoxOffice"])

df["Rated"] = df["Rated"].str.upper().str.strip()
df["Genre"] = df["Genre"].str.split(",").str[0].str.lower()

FEATURES = [
    "mean_sent",
    "std_sent",
    "Year",
    "Runtime",
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Rotten Tomatoes",
    "Metacritic",
    "Rated",
    "Genre",
]
X = df[FEATURES]
y = df["Log_BoxOffice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Preprocessing pipeline (OHE now dense)
numeric_features = [
    "mean_sent",
    "std_sent",
    "Year",
    "Runtime",
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Rotten Tomatoes",
    "Metacritic",
]
categorical_features = ["Rated", "Genre"]

numeric_pipe = SimpleImputer(strategy="mean")
categorical_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)

rf_pipe = Pipeline(
    [("prep", preprocessor), ("rf", RandomForestRegressor(random_state=42))]
)

# 3) Permutation importance on original FEATURES
rf_pipe.fit(X_train, y_train)
perm = permutation_importance(
    rf_pipe, X_test, y_test, n_repeats=10, random_state=0, scoring="r2"
)

perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print("=== Permutation Importances (R² drop) ===")
print(perm_imp.head(10), "\n")

# 4) SHAP global explanation (fallback to KernelExplainer)

# Precompute the dense, preprocessed data
X_train_p = preprocessor.transform(X_train)
X_test_p = preprocessor.transform(X_test[:200])

# Build the full feature names list for plotting
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
cat_ohe_names = list(ohe.get_feature_names_out(categorical_features))
shap_feature_names = numeric_features + cat_ohe_names

# Use a small background sample for speed
background = shap.sample(X_train_p, 50, random_state=42)

# Create a KernelExplainer around the RF predict function
explainer = shap.KernelExplainer(rf_pipe.named_steps["rf"].predict, background)
# Compute SHAP values for our test subset
shap_vals = explainer.shap_values(X_test_p)

print("=== SHAP summary plot upcoming ===")
shap.summary_plot(
    shap_vals, features=X_test_p, feature_names=shap_feature_names, show=False
)


# 5) Hyperparameter tuning
param_dist = {
    "rf__n_estimators": [100, 300, 500],
    "rf__max_depth": [None, 10, 30, 50],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", "log2", 0.5],
}
rand_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_
print("=== Best RF hyperparameters ===")
print(rand_search.best_params_, "\n")

# 6) Residual analysis by genre
y_pred = best_rf.predict(X_test)
residuals = np.expm1(y_test) - np.expm1(y_pred)

res_df = X_test.copy()
res_df["residual"] = residuals
res_by_genre = (
    res_df.groupby("Genre")["residual"].agg(["mean", "count"]).sort_values("mean")
)
print("=== Mean residual ($) by Genre ===")
print(res_by_genre.head(), "\n")

# 7) Cross‐validation of tuned model
cv = KFold(5, shuffle=True, random_state=0)
cv_scores = cross_val_score(best_rf, X, y, cv=cv, scoring="r2", n_jobs=-1)
print(f"5-fold CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}\n")

with open("BestRF_pipeline.pkl", "wb") as f:
    pickle.dump(best_rf, f)

print("Complete: permutation importance, SHAP, tuning, residuals, CV, and saved model.")
