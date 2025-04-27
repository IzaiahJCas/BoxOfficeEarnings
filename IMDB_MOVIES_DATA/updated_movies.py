import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load your original movie list
movie_df = pd.read_csv("filtered_imdb_movies.csv")

# Find the index where "Devil's Pass" is
devils_pass_index = movie_df[movie_df["title"] == "Devil's Pass"].index

if len(devils_pass_index) == 0:
    print("Devil's Pass not found in your movie list.")
else:
    cut_index = devils_pass_index[0] + 1  # Move one after Devil's Pass
    remaining_movies_df = movie_df.iloc[cut_index:]

    # Save to a new file
    remaining_movies_df.to_csv("remaining_movies_after_devils_pass.csv", index=False)
    print(
        f"Saved {len(remaining_movies_df)} remaining movies to 'remaining_movies_after_devils_pass.csv'."
    )
