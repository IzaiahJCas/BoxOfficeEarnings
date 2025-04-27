import dask.dataframe as dd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dask import delayed
from tqdm import tqdm
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load cleaned Reddit dataset
df = dd.read_csv("cleaned_reddit_sentiment.csv")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


# Define a non-delayed version (because Dask will handle partitions)
def compute_sentiment_batch(part):
    tqdm.pandas(
        desc="Sentiment Progress"
    )  # Create tqdm progress bar inside each partition
    return part.progress_apply(lambda x: analyzer.polarity_scores(x)["compound"])


# Apply VADER + progress bar across partitions
sentiment_series = df["comment_text"].map_partitions(
    compute_sentiment_batch, meta=("comment_sentiment", "float64")
)

# Add the sentiment column
df["comment_sentiment"] = sentiment_series

# Save the result
df.to_csv("reddit_with_sentiment.csv", single_file=True, index=False)

print("Done computing sentiment scores and saved as reddit_with_sentiment.csv")
