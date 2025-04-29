import pandas as pd
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load your full filtered IMDb movies list
full_movies_df = pd.read_csv("filtered_imdb_movies.csv")

# 2. Load distinct movies you already scraped
scraped_movies_df = pd.read_csv("distinct_movie_titles.csv")

# 3. Load movies already scheduled in remaining batch
remaining_movies_df = pd.read_csv("remaining_movies.csv")

scrape1 = pd.read_csv("combined_scrape1.csv")
scrape2 = pd.read_csv("combined_scrape2.csv")
scrape3 = pd.read_csv("reddit_sentiment_remaining_checkpoint_3900.csv")

# 5. Combine all titles you have already scraped or scheduled
already_done_titles = pd.concat(
    [
        scraped_movies_df["movie_title"],
        remaining_movies_df["title"],
        scrape1["movie_title"],
        scrape2["movie_title"],
        scrape3["movie_title"],
    ],
).drop_duplicates()

# 6. Filter: Keep only movies NOT in already_done_titles
filtered_movies_df = full_movies_df[~full_movies_df["title"].isin(already_done_titles)]

# 7. Save the result
filtered_movies_df.to_csv("movies_to_scrape_next.csv", index=False)

print(
    f"Filtered list saved! {len(filtered_movies_df)} movies still need to be scraped."
)
