import pandas as pd
import os

# Set working directory (if needed)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the two datasets
movies_df = pd.read_csv("final_movies_cleaned.csv")
sentiment_df = pd.read_csv("aggregated_movie_sentiment.csv")

# 3. Merge datasets on movie title
merged_df = pd.merge(
    movies_df, sentiment_df, left_on="Title", right_on="movie_title", how="inner"
)

# 5. Drop rows with missing values (if any)
merged_df = merged_df.dropna()

# 6. Save merged data
merged_df.to_csv("merged_movie_data.csv", index=False)

print("Merged movies and sentiment data saved as merged_movie_data.csv")
