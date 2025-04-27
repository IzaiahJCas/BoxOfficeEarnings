import os
import pandas as pd
import ast

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Step 1: Load the original CSV
df = pd.read_csv("omdb_movies.csv")


# Step 2: Function to safely extract a specific rating
def extract_specific_rating(ratings_str, source_name):
    try:
        ratings_list = ast.literal_eval(ratings_str)
        if isinstance(ratings_list, list):
            for rating in ratings_list:
                if rating.get("Source") == source_name:
                    return rating.get("Value")
    except (ValueError, SyntaxError):
        pass
    return None


# Step 3: Create new columns for the three sources
df["Internet Movie Database"] = df["Ratings"].apply(
    lambda x: extract_specific_rating(x, "Internet Movie Database")
)
df["Rotten Tomatoes"] = df["Ratings"].apply(
    lambda x: extract_specific_rating(x, "Rotten Tomatoes")
)
df["Metacritic"] = df["Ratings"].apply(
    lambda x: extract_specific_rating(x, "Metacritic")
)


# Step 4: Clean the extracted ratings (remove '/10', '/100', etc.)
def clean_rating_value(value):
    if isinstance(value, str):
        if "/" in value:
            value = value.split("/")[0]  # Keep the part before '/'
        value = value.strip()  # Remove any leading/trailing spaces
    return value


for col in ["Internet Movie Database", "Rotten Tomatoes", "Metacritic"]:
    df[col] = df[col].apply(clean_rating_value)

# Step 5: Drop the 'Ratings' column and 'DVD' column
df.drop(columns=["Ratings", "DVD"], inplace=True)

# Step 6: Save the cleaned file
df.to_csv("final_movies.csv", index=False)

print("Cleaning completed! Saved as 'final_cleaned_movies.csv'.")
