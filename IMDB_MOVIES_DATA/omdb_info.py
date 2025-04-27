import pandas as pd
import requests
import time
from dotenv import load_dotenv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load your OMDb API key from .env file
load_dotenv()
OMDB_API_KEY = os.getenv("omdb_api_key")

if OMDB_API_KEY is None:
    raise ValueError(
        "OMDb API key not found! Please set 'omdb_api_key' in your .env file."
    )

# Load your filtered IMDb movies list
df = pd.read_csv("filtered_imdb_movies.csv")

# Prepare a list to hold only OMDb fields
omdb_data = []

# Loop through each movie by IMDb ID
for idx, row in df.iterrows():
    imdb_id = row["id"]  # 'id' column is IMDb ID

    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("Response") == "True":
        # Only save new OMDb data (no old columns at all)
        omdb_movie = {
            "imdbID": data.get("imdbID", None),
            "Title": data.get("Title", None),
            "Year": data.get("Year", None),
            "Rated": data.get("Rated", None),
            "Released": data.get("Released", None),
            "Runtime": data.get("Runtime", None),
            "Genre": data.get("Genre", None),
            "Director": data.get("Director", None),
            "Writer": data.get("Writer", None),
            "Actors": data.get("Actors", None),
            "Plot": data.get("Plot", None),
            "Language": data.get("Language", None),
            "Country": data.get("Country", None),
            "Awards": data.get("Awards", None),
            "Ratings": data.get("Ratings", None),
            "Metascore": data.get("Metascore", None),
            "imdbRating": data.get("imdbRating", None),
            "imdbVotes": data.get("imdbVotes", None),
            "BoxOffice": data.get("BoxOffice", None),
            "DVD": data.get("DVD", None),
        }
        omdb_data.append(omdb_movie)
        print(f"Fetched OMDb data for {data.get('Title', imdb_id)}")
    else:
        print(f"Failed to fetch OMDb data for IMDb ID: {imdb_id}")

    # Sleep to avoid rate limits
    time.sleep(0.5)

# Convert collected OMDb data to DataFrame
omdb_df = pd.DataFrame(omdb_data)

# Save to new CSV
omdb_df.to_csv("omdb_enriched_movies.csv", index=False)

print(f"Done! Saved {len(omdb_df)} movies into 'omdb_enriched_movies.csv'.")
