import pandas as pd

# Data collection for box office revenue
# The data came from the-numbers.com
data = {
    'movie_title': ['Barbie', 'Oppenheimer', 'Avatar The Way of Water', 'The Batman', 
                    'Black Panther Wakanda Forever', 'Dune Part Two', 'John Wick Chapter 4', 
                    'Nope', 'The Super Mario Bros Movie', 'Top Gun Maverick'],
    'Domestic Box Office': [636785476, 330078895, 684075767, 369612903, 453829060, 282709065, 187131806, 123277080, 574934330, 718732821],
    'International Box Office': [807673753, 646185644, 1636174514, 401247477, 405379776, 432002455, 253188772, 48762007, 784212298, 735268289],
    'Worldwide Box Office': [1444459229, 976264539, 2320250281, 770860380, 859208836, 714711520, 440320578, 172039087, 1359146628, 1454001110]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('box_office_revenue.csv', index=False)

print("CSV file created successfully with box office data.")

# Load both CSV files
reddit_summary = pd.read_csv('reddit_movie_summary_trial.csv')
box_office = pd.read_csv('box_office_revenue.csv')

# Merge on movie_title - now that columns match
merged = pd.merge(reddit_summary, box_office, on='movie_title', how='left')

# Print the merged result
print(merged.head())

# Optionally, save the merged result
merged.to_csv("reddit_movie_summary_with_revenue.csv", index=False)
