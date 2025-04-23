import praw
import pandas as pd
from dotenv import load_dotenv
import os
import datetime

# Load credentials from .env
load_dotenv()

# Authenticate with Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 by u/csds312_groupProject"
)

def search_reddit_to_csv(subreddit_name, keyWords, sortBy, postLimit, commentAmount, commentLength, output_csv):
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    for post in subreddit.search(keyWords, sort=sortBy, limit=postLimit):
        post.comments.replace_more(limit=0)  # Get all top-level comments

        for comment in post.comments[:commentAmount]:
            data.append({
                "post_title": post.title,
                "post_score": post.score,
                "post_url": post.url,
                "subreddit": str(post.subreddit),
                "comment_text": comment.body[:commentLength],
                "comment_score": comment.score,
                "created_utc": datetime.datetime.fromtimestamp(comment.created_utc)
            })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} comments to {output_csv}")

# Example usage
search_reddit_to_csv(
    subreddit_name="movies",
    keyWords="minecraft movie",
    sortBy="relevance",
    postLimit=10,
    commentAmount=3,
    commentLength=150,
    output_csv="reddit_minecraft_comments.csv"
)
