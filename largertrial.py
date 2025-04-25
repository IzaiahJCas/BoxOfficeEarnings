import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
import datetime
import matplotlib.pyplot as plt

# Load Reddit API credentials
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 trial by u/csds312_groupProject"
)

# Movies to analyze
movies = []
with open('Movie_Titles/action.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        movies.append(row[0])  # Assuming title is the first column

analyzer = SentimentIntensityAnalyzer()
all_data = []

# Scrape Reddit and analyze sentiment
for movie in movies:
    subreddit = reddit.subreddit("movies")
    for post in subreddit.search(movie, sort="relevance", limit=5):
        post.comments.replace_more(limit=0)
        for comment in post.comments[:5]:  # Take 5 comments per post
            sentiment = analyzer.polarity_scores(comment.body)
            all_data.append({
                "movie_title": movie,
                "post_title": post.title,
                "comment_text": comment.body[:200],
                "sentiment_compound": sentiment["compound"],
                "comment_score": comment.score,
                "created_utc": datetime.datetime.fromtimestamp(comment.created_utc)
            })

# Create DataFrame
df = pd.DataFrame(all_data)

# Aggregate per movie
summary = df.groupby("movie_title").agg({
    "sentiment_compound": "mean",
    "comment_text": "count",
    "comment_score": "mean"
}).reset_index()

summary.rename(columns={
    "sentiment_compound": "avg_sentiment",
    "comment_text": "num_comments",
    "comment_score": "avg_comment_score"
}, inplace=True)

print(summary)

# Save to CSV
df.to_csv("reddit_movie_comments_trial.csv", index=False)
summary.to_csv("reddit_movie_summary_trial.csv", index=False)

# EDA Plot 1: Average Sentiment per Movie
plt.figure(figsize=(10, 5))
plt.bar(summary["movie_title"], summary["avg_sentiment"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Sentiment")
plt.title("Average Reddit Sentiment per Movie")
plt.tight_layout()
plt.show()
