import dask.dataframe as dd
import dask
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Step 1: Force columns to be loaded as text/object first
df = dd.read_csv(
    "reddit_sentiment_checkpoint_7000.csv",
    dtype={"post_score": "object", "comment_score": "object"},
    assume_missing=True,  # Important for mixed type columns
)

# Step 2: Remove '[deleted]' and '[removed]' comments
df = df[~df["comment_text"].isin(["[deleted]", "[removed]"])]

# Step 3: Drop duplicates
df = df.drop_duplicates()

# Step 4: Strip comment text
df["comment_text"] = df["comment_text"].str.strip()

# Step 5: Drop rows with missing or empty comments
df = df.dropna(subset=["comment_text"])
df = df[df["comment_text"].str.len() > 0]

# Step 6: Convert created_utc to datetime
df["created_utc"] = dd.to_datetime(df["created_utc"], errors="coerce")


# Step 7: Clean post_score and comment_score
# - Try to convert to float safely
# - If fail, set to 0
def safe_to_int(val):
    try:
        return int(float(val))
    except:
        return 0


df["post_score"] = df["post_score"].map(safe_to_int, meta=("post_score", "int64"))
df["comment_score"] = df["comment_score"].map(
    safe_to_int, meta=("comment_score", "int64")
)

# Step 8: Drop rows missing important fields
df = df.dropna(subset=["movie_title"])

# Step 9: Save cleaned file
df.to_csv("cleaned_reddit_sentiment.csv", single_file=True, index=False)

print("Done cleaning reddit_sentiment_checkpoint_7000.csv with safe type handling")
