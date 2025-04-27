import dask.dataframe as dd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the Reddit data with sentiment scores
df = dd.read_csv("reddit_with_sentiment.csv")

# Group by movie_title and compute mean sentiment
agg_df = df.groupby("movie_title").comment_sentiment.mean().reset_index()

# Rename column for clarity
agg_df = agg_df.rename(columns={"comment_sentiment": "mean_comment_sentiment"})

# Save aggregated dataset
agg_df.to_csv("aggregated_movie_sentiment.csv", single_file=True, index=False)

print("Aggregated mean sentiment score per movie and saved.")
