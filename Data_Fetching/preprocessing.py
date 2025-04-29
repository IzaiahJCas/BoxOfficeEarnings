import pandas as pd
import csv
import os

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
        print(f"Trying to load {file}...")
        good_rows = []
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_columns = len(header)
            for row in reader:
                if len(row) == expected_columns:
                    good_rows.append(row)
                else:
                    # Skip bad row
                    pass

        if good_rows:
            df = pd.DataFrame(good_rows, columns=header)
            good_dataframes.append(df)
            print(f"{file} loaded with {len(df)} rows.")
        else:
            print(f"{file} had no good rows.")
    except Exception as e:
        print(f"Failed to clean {file}: {e}")

# Merge everything together
if good_dataframes:
    merged_df = pd.concat(good_dataframes, ignore_index=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.to_csv("final_data_cleaned.csv", index=False)
    print(f"Merged {len(csv_files)} files successfully into 'final_data_cleaned.csv'.")
    print(f"Final shape: {merged_df.shape}")
else:
    print("No valid dataframes to merge.")
