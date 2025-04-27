import pandas as pd
from supabase import create_client, Client
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
SUPABASE_URL = "https://flpnzndwioffpeyozign.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZscG56bmR3aW9mZnBleW96aWduIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUzNzAyOTEsImV4cCI6MjA2MDk0NjI5MX0.bEIk3wA28khyHaa2hw6BhtX3MHPVKLL57qX3HSCu8E0"  # must be service_role key not anon key

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Load data
df = pd.read_csv("reddit_sentiment_checkpoint_7000.csv", encoding="utf-8")


def safe_text(x):
    try:
        return x.encode("utf-8", "ignore").decode("utf-8")
    except:
        return ""


text_columns = ["movie_title", "post_title", "post_url", "subreddit", "comment_text"]
for col in text_columns:
    df[col] = df[col].astype(str).apply(safe_text)

print(f"Loaded {len(df)} rows.")

batch_size = 500

for i in range(0, len(df), batch_size):
    batch = df.iloc[i : i + batch_size]
    records = batch.to_dict(orient="records")

    try:
        response = supabase.table("reddit_sentiments").insert(records).execute()
        print(f"Batch {i//batch_size + 1} inserted. Response: {response}")
    except Exception as e:
        print(f"Error at batch {i//batch_size + 1}: {e}")
