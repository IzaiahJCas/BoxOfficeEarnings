import pandas as pd

# Load full IMDB dataset
df = pd.read_csv("imdb_movies.csv")

# Apply the filter
filtered_df = df[
    (df["type"] == "movie")
    & (df["releaseYear"] >= 2000)
    & (df["releaseYear"] <= 2025)
    & (df["numVotes"] > 5000)
]

# Save the full filtered data (all columns) to a new CSV
filtered_df.to_csv("filtered_imdb_movies.csv", index=False)

print(f"✅ Saved {len(filtered_df)} movies to 'filtered_imdb_movies.csv'")
