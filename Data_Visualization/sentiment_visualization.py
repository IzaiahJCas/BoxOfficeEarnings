import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Create a folder to save plots
os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("reddit_with_sentiment.csv")

# Preview
print(df.head())
print(df.info())

# 1. Sentiment Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["comment_sentiment"], bins=100, kde=True)
plt.title("Distribution of Comment Sentiment Scores")
plt.xlabel("Comment Sentiment Score")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("plots/sentiment_distribution.png")
plt.close()

# 2. Boxplot of Sentiment Scores
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["comment_sentiment"])
plt.title("Boxplot of Comment Sentiment Scores")
plt.xlabel("Comment Sentiment Score")
plt.grid(True)
plt.savefig("plots/boxplot_sentiment.png")
plt.close()

# 3. Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df["comment_sentiment"], fill=True)
plt.title("Density Plot of Comment Sentiment Scores")
plt.xlabel("Comment Sentiment Score")
plt.grid(True)
plt.savefig("plots/density_sentiment.png")
plt.close()

# 4. Positive vs Negative Comments
df["sentiment_category"] = df["comment_sentiment"].apply(
    lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
)
sentiment_counts = df["sentiment_category"].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title("Counts of Positive, Negative, Neutral Comments")
plt.ylabel("Number of Comments")
plt.xlabel("Sentiment Category")
plt.grid(True)
plt.savefig("plots/pos_neg_neutral_counts.png")
plt.close()

# 5. Comment Length vs Sentiment Score
df["comment_length"] = df["comment_text"].astype(str).apply(len)

plt.figure(figsize=(10, 6))
sns.scatterplot(x="comment_length", y="comment_sentiment", data=df, alpha=0.3)
plt.title("Comment Length vs Comment Sentiment Score")
plt.xlabel("Comment Length (characters)")
plt.ylabel("Comment Sentiment Score")
plt.grid(True)
plt.savefig("plots/comment_length_vs_sentiment.png")
plt.close()

# 6. Sentiment Over Time (if timestamps exist)
if "created_utc" in df.columns:
    # Only convert if not already datetime
    if not np.issubdtype(df["created_utc"].dtype, np.datetime64):
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    df.set_index("created_utc", inplace=True)

    plt.figure(figsize=(14, 6))
    df["comment_sentiment"].resample("D").mean().plot()
    plt.title("Average Comment Sentiment Score Over Time")
    plt.ylabel("Average Sentiment Score")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig("plots/sentiment_over_time.png")
    plt.close()

# 7. Top 5 Positive and Negative Comments
top_positive = df.sort_values(by="comment_sentiment", ascending=False).head(5)[
    ["comment_text", "comment_sentiment"]
]
top_negative = df.sort_values(by="comment_sentiment", ascending=True).head(5)[
    ["comment_text", "comment_sentiment"]
]

print("\nTop 5 Positive Comments:")
print(top_positive.to_string(index=False))

print("\nTop 5 Negative Comments:")
print(top_negative.to_string(index=False))

# 8. Sentiment by Movie
if "movie_title" in df.columns:
    movie_sentiment = (
        df.groupby("movie_title")["comment_sentiment"].mean().sort_values()
    )

    plt.figure(figsize=(14, 8))
    movie_sentiment.tail(20).plot(kind="barh")
    plt.title("Top 20 Movies by Highest Average Sentiment")
    plt.xlabel("Average Sentiment Score")
    plt.grid(True)
    plt.savefig("plots/top_20_highest_sentiment_movies.png")
    plt.close()

    plt.figure(figsize=(14, 8))
    movie_sentiment.head(20).plot(kind="barh", color="red")
    plt.title("Top 20 Movies by Lowest Average Sentiment")
    plt.xlabel("Average Sentiment Score")
    plt.grid(True)
    plt.savefig("plots/top_20_lowest_sentiment_movies.png")
    plt.close()
