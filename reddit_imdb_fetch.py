import praw
import pandas as pd
from dotenv import load_dotenv
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load credentials
load_dotenv()

# Authenticate
reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 by u/csds312_groupProject",
)

# Load IMDB movie titles
df = pd.read_csv("imdb_movies.csv")
filtered = df[
    (df["type"] == "movie") & (df["releaseYear"] >= 2000) & (df["releaseYear"] <= 2025)
]
top_titles = filtered[filtered["numVotes"] > 5000]["title"].dropna().unique().tolist()


# Fetch Reddit comments for one title
def fetch_comments_for_title(title):
    subreddit = reddit.subreddit("movies")
    rows = []

    try:
        for post in subreddit.search(
            title, sort="relevance", limit=10
        ):  # increase posts per movie
            post.comments.replace_more(limit=0)
            for comment in post.comments[:5]:  # get 5 comments per post
                rows.append(
                    {
                        "movie_title": title,
                        "post_title": post.title,
                        "post_score": post.score,
                        "post_url": post.url,
                        "subreddit": str(post.subreddit),
                        "comment_text": comment.body[:150],
                        "comment_score": comment.score,
                        "created_utc": datetime.datetime.fromtimestamp(
                            comment.created_utc
                        ),
                    }
                )
    except Exception as e:
        print(f"❌ Error for {title}: {e}")
    return rows


# Parallel fetching
all_data = []
max_threads = 10  # careful: don't use too many threads or you will get rate limited
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(fetch_comments_for_title, title) for title in top_titles]
    for future in as_completed(futures):
        result = future.result()
        if result:
            all_data.extend(result)

# Save
df_combined = pd.DataFrame(all_data)
df_combined.to_csv("combined_reddit_sentiment.csv", index=False)
print(f"✅ Saved {len(df_combined)} total comments.")
