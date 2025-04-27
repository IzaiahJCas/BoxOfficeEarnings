import pandas as pd
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the Reddit data with sentiment
df = pd.read_csv("reddit_with_sentiment.csv")

# Drop rows where movie_title is missing
df = df.dropna(subset=["movie_title"])

# Count distinct movie titles
distinct_movies = df["movie_title"].nunique()

# Get all unique movie titles
unique_titles_list = df["movie_title"].drop_duplicates().tolist()

# Print summary
print(f"Number of distinct movie titles in Reddit data: {distinct_movies}")
print("\nSample of unique movie titles:")
print(unique_titles_list[:10])  # print first 10 for sample

# Save full list to a CSV file
output_df = pd.DataFrame(unique_titles_list, columns=["movie_title"])
output_df.to_csv("distinct_movie_titles.csv", index=False)

print("\nFull list of distinct movie titles saved to distinct_movie_titles.csv!")
