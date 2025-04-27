import pandas as pd
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the filtered movies to scrape
movies_df = pd.read_csv("movies_to_scrape_next.csv")

# 3. Split into 2 parts
half = len(movies_df) // 2

movies_part1 = movies_df.iloc[:half]
movies_part2 = movies_df.iloc[half:]

# 4. Save each part
movies_part1.to_csv("movies_to_scrape_part1.csv", index=False)
movies_part2.to_csv("movies_to_scrape_part2.csv", index=False)

print(f"    Dataset split into 2 parts:")
print(f"    Part 1: {len(movies_part1)} movies → movies_to_scrape_part1.csv")
print(f"    Part 2: {len(movies_part2)} movies → movies_to_scrape_part2.csv")
