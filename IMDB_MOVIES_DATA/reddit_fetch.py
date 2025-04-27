import praw
import pandas as pd
from dotenv import load_dotenv
import os
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load credentials
load_dotenv()
print("Script started", flush=True)


# Authenticate

reddit = praw.Reddit(
    client_id=os.getenv("clientID"),
    client_secret=os.getenv("clientSecret"),
    user_agent="csds312 by u/csds312_groupProject",
)

print("Reddit client authenticated", flush=True)


# Load IMDB movie titles

df = pd.read_csv("imdb_movies.csv")

filtered = df[
    (df["type"] == "movie") & (df["releaseYear"] >= 2000) & (df["releaseYear"] <= 2025)
]

top_titles = filtered[filtered["numVotes"] > 5000]["title"].dropna().unique().tolist()

print(f"Loaded {len(top_titles)} movie titles", flush=True)


# Fetch Reddit comments for one title


def fetch_comments_for_title(title):

    subreddit = reddit.subreddit("movies")

    rows = []

    try:

        for post in subreddit.search(title, sort="relevance", limit=10):

            post.comments.replace_more(limit=0)

            for comment in post.comments[:20]:

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

        print(f"Finished scraping: {title} at {datetime.datetime.now()}", flush=True)

    except Exception as e:

        print(f"Error for {title}: {e}", flush=True)

    return rows


# Parallel fetching

all_data = []

max_threads = 10

with ThreadPoolExecutor(max_workers=max_threads) as executor:

    futures = [executor.submit(fetch_comments_for_title, title) for title in top_titles]

    for idx, future in enumerate(as_completed(futures), 1):

        result = future.result()

        if result:

            all_data.extend(result)

        # Save intermediate every 100 movies

        if idx % 100 == 0:

            checkpoint_file = f"reddit_sentiment_checkpoint_{idx}.csv"

            pd.DataFrame(all_data).to_csv(checkpoint_file, index=False)

            print(f"Saved checkpoint after {idx} movies: {checkpoint_file}", flush=True)


# Final save

df_combined = pd.DataFrame(all_data)

df_combined.to_csv("combined_reddit_sentiment.csv", index=False)
print(f"All done! Saved {len(df_combined)} total comments.", flush=True)
