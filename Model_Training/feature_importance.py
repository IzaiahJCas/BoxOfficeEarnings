import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 1. Load the trained Random Forest model
with open("RandomForestRegressor.pkl", "rb") as f:
    rf_model = pickle.load(f)

# 2. List of original feature names
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

# 3. After preprocessing, OneHotEncoder expands categorical features
# So we need to get the feature names properly
# Extract feature names from the onehot encoder
ohe = (
    rf_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)
ohe_feature_names = ohe.get_feature_names_out(categorical_features)

# Final full feature list
final_feature_names = numeric_features + list(ohe_feature_names)

# 4. Get feature importances
importances = rf_model.named_steps["regressor"].feature_importances_

# 5. Create a DataFrame for easier handling
feat_importances_df = pd.DataFrame(
    {"feature": final_feature_names, "importance": importances}
).sort_values(by="importance", ascending=False)

print("Feature importances:")
print(feat_importances_df)

# 6. Plot nicely
plt.figure(figsize=(10, 6))
plt.barh(feat_importances_df["feature"], feat_importances_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
