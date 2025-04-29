import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import csv
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load cleaned file
df = pd.read_csv(
    "final_data_cleaned.csv",
    quoting=1,  # Safe loading
    on_bad_lines="skip",
    encoding_errors="replace",
)

print(f"Loaded {len(df)} rows for sentiment analysis.")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sentiment calculation
tqdm.pandas(desc="Sentiment Progress")

df["comment_sentiment"] = df["comment_text"].progress_apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"] if pd.notnull(x) else 0.0
)


# Clean comment_text from dangerous characters
def clean_comment_text(text):
    if pd.isnull(text):
        return text
    text = str(text)
    text = text.replace('"', "'")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text


print("Cleaning comment_text before saving...")
df["comment_text"] = df["comment_text"].apply(clean_comment_text)

# Save output
df.to_csv("reddit_with_sentiment.csv", index=False, quoting=csv.QUOTE_ALL)

print("Done saving reddit_with_sentiment.csv with sentiment scores.")
