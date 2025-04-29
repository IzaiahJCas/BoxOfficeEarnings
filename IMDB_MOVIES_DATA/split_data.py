import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
chunk_size = 49000
csv_file = "reddit_sentiment_checkpoint_7000.csv"
for idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
    chunk.to_csv(f"combined_reddit_sentiment_part_{idx}.csv", index=False)
