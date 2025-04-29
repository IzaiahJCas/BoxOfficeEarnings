import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the dataset
movies = pd.read_csv(r'MoviesCleaned_rows.csv')

# Display basic info
print("\nDataset Info:")
print(movies.info())

print("\nMissing Values:")
print(movies.isnull().sum())

print("\nSummary Statistics:")
print(movies.describe(include='all'))

# Set general style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

#  Missing Data Visualization
# Helps identify data quality problems. Missing values can affect modeling. You might want to drop/fill/ignore columns with lots of missing data
plt.figure(figsize=(12,6))
sns.heatmap(movies.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Movies', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

# Distribution Plots 
# Distribution of BoxOffice: Shows that most movies make low amounts and only a few movies earn a lot. Important for understanding the target if you're predicting revenue.
# Distribution of IMDb rating: Most ratings are between 5–7.5. Indicates that very bad or very great ratings are rare.
# Distribution of Rotten Tomatoes: Shows how critics rated movies.  Helps understand critic bias.
# Distribution of Runtime: Most movies are between 90 and 120 minutes. Outliers (very short or long movies) could be special cases like short films or director’s cuts.
features_to_plot = ['BoxOffice', 'imdbRating', 'Rotten Tomatoes', 'Runtime']

for feature in features_to_plot:
    plt.figure(figsize=(10,6))
    sns.histplot(movies[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(f'{feature}', fontsize=14)
    plt.ylabel('Number of Movies', fontsize=14)
    plt.grid(True)
    plt.show()

# Distribution of BoxOffice (log scale because skewed)
plt.figure(figsize=(10,6))
sns.histplot(
    movies['BoxOffice'],
    kde=True,
    bins=30, # due to log skew the histogram bars are very small
    log_scale=(False, True)
)
plt.title('Distribution of BoxOffice (Log Scaled Y-Axis)', fontsize=16)
plt.xlabel('Box Office Earnings in Millions (USD)', fontsize=14)
plt.ylabel('Number of Movies (Log Scale)', fontsize=14)
plt.grid(True)
plt.show()

# Correlation Heatmap
# Metascore, Rotten Tomatoes, and IMDb ratings are fairly correlated (good movies get good reviews everywhere)
# Box Office is only weakly correlated with ratings. High rating does not guarantee high revenue. This gives insight for modeling (BoxOffice prediction won't just depend on ratings).
numeric_features = [
    'Year', 'Runtime', 'Metascore', 'imdbRating',
    'imdbVotes', 'BoxOffice', 'Internet Movie Database',
    'Rotten Tomatoes', 'Metacritic'
]
corr_matrix = movies[numeric_features].corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Top Genres 
# Most common genres among movies
genre_counter = Counter()
for genres in movies['Genre']:
    for genre in genres.split(','):
        genre_counter[genre.strip()] += 1

# Top 10 genres
top_genres = genre_counter.most_common(10)
genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

plt.figure(figsize=(10,6))
sns.barplot(data=genres_df, x='Count', y='Genre')
plt.title('Top 10 Movie Genres by Count', fontsize=16)
plt.xlabel('Number of Movies', fontsize=14)
plt.ylabel('Genre', fontsize=14)
plt.grid(True)
plt.show()

# Top Directors 
# Directors with the most movies
top_directors = movies['Director'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title('Top 10 Directors by Number of Movies', fontsize=16)
plt.xlabel('Number of Movies Directed', fontsize=14)
plt.ylabel('Director', fontsize=14)
plt.grid(True)
plt.show()

# Relationship between IMDb Rating and BoxOffice 
# Relationship between movie quality and earnings
plt.figure(figsize=(10,6))
sns.scatterplot(x='imdbRating', y='BoxOffice', data=movies)
plt.title('Relationship between IMDb Rating and Box Office Earnings', fontsize=16)
plt.xlabel('IMDb Rating (out of 10)', fontsize=14)
plt.ylabel('Box Office Earnings in Millions (USD)', fontsize=14)
plt.grid(True)
plt.show()
