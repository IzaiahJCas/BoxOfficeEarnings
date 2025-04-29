import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import numpy as np

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the merged dataset
merged_df = pd.read_csv("merged_movie_data.csv")

# Log transform BoxOffice
merged_df["Log_BoxOffice"] = np.log1p(
    merged_df["BoxOffice"]
)  # log(1+x) handles 0 safely

# # Convert runtime to numeric minutes
# merged_df["Runtime"] = merged_df["Runtime"].str.extract("(\d+)").astype(float)

# Normalize categorical fields
merged_df["Rated"] = merged_df["Rated"].str.upper().str.strip()
merged_df["Genre"] = (
    merged_df["Genre"].str.split(",").str[0].str.strip().str.lower()
)  # First genre only

# Fill missing values for sentiment and num_comments if needed (optional, should check if missing first)
merged_df["mean_comment_sentiment"] = merged_df["mean_comment_sentiment"].fillna(0)

# 3. Prepare features and labels
features = [
    "mean_comment_sentiment",
    "Year",
    "Runtime",
    "Rated",
    "Genre",
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Rotten Tomatoes",
    "Metacritic",
]

X = merged_df[features]
y = merged_df["Log_BoxOffice"]

# 4. Preprocessing pipeline
# Define which columns are numeric vs categorical
numeric_features = [
    "mean_comment_sentiment",
    "Year",
    "Runtime",
    "Metascore",
    "imdbRating",
    "imdbVotes",
    "Rotten Tomatoes",
    "Metacritic",
]
categorical_features = ["Rated", "Genre"]

# Pipelines for numeric and categorical data
numeric_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Full preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Initialize models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}

# 7. Train, predict, evaluate each model
results = {}

for name, base_model in models.items():
    print(f"Training {name}...")

    # Create a pipeline: preprocessor + model
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", base_model)])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Undo log1p to get back real BoxOffice predictions
    y_pred_real = np.expm1(y_pred)
    y_test_real = np.expm1(y_test)

    # Evaluation metrics
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

    # Save the pipeline (preprocessing + model together)
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"    {name} Results:")
    print(f"    R² Score: {r2:.4f}")
    print(f"    Mean Absolute Error (MAE): {mae:.2f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.2f}")
    print()

# 8. Summarize all results
print("All Model Results:")
for name, metrics in results.items():
    print(
        f"{name}: R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.2f}, RMSE = {metrics['RMSE']:.2f}"
    )

print("\nAll models trained, evaluated, and saved!")
