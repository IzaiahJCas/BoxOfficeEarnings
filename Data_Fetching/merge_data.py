import pandas as pd
import os
import csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of CSV files to merge
csv_files = [
    "cleaned_reddit_sentiment.csv",
    "scrape1_cleaned.csv",
    "scrape2_cleaned.csv",
    "scrape3_cleaned.csv",
]

good_dataframes = []

for file in csv_files:
    try:
        print(f"Loading {file}...")
        df = pd.read_csv(
            file,
            quoting=csv.QUOTE_NONE,  # Ignore quote errors
            on_bad_lines="skip",  # Skip broken rows
            encoding_errors="replace",  # Fix weird characters
            low_memory=False,
        )
        good_dataframes.append(df)
    except Exception as e:
        print(f"Failed to load {file}: {e}")

# Concatenate everything
merged_df = pd.concat(good_dataframes, ignore_index=True)

# Drop exact duplicates
merged_df.drop_duplicates(inplace=True)

# Save the clean merged file
merged_df.to_csv("final_data_cleaned.csv", index=False)

print(f"Successfully merged {len(csv_files)} files into 'final_data_cleaned.csv'.")
print(f"Final shape: {merged_df.shape}")
