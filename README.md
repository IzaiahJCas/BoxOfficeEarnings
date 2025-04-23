# BoxOfficeEarnings
Group project for CSDS 312

reddit.py scrapes reddit data and puts it into our Supabase database. It will take a .csv file as an input:
the .csv file will be in the format {movie_title,year,genre}. These files are stored in the folder "Movie_Titles"
If you want to scrape more movies, just add a .csv file in a similar format and run it through reddit.py, and the Supabase DB will be populated.
