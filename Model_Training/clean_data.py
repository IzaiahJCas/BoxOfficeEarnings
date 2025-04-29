import csv
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
good_rows = []

with open("final_data.csv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    header = next(reader)
    expected_columns = len(header)

    for row in reader:
        if len(row) == expected_columns:
            good_rows.append(row)
        else:
            # Skip broken row
            pass

# Save a truly clean CSV
df_clean = pd.DataFrame(good_rows, columns=header)
df_clean.to_csv("final_data_cleaned.csv", index=False)

print(f"Saved final clean file with {len(df_clean)} rows.")
