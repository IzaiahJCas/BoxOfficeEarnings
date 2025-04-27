from supabase import create_client, Client
url = "https://flpnzndwioffpeyozign.supabase.co"
with open('SupabaseKey.txt', 'r') as file:
    key = file.read().strip()
supabase: Client = create_client(url, key)

reddit_data = supabase.table('reddit_sentiments').select('movie_title, comment_text').execute()
print("Table 1 loaded")

movies_data = supabase.table('MoviesCleaned').select('Title, BoxOffice').execute()
print("Table 2 loaded")

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
reddit_df = pd.DataFrame(reddit_data.data)
reddit_df['comment_sentiment'] = reddit_df['comment_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
movies_df = pd.DataFrame(movies_data.data)
print("Pandas df loaded")
print(reddit_df)
print(movies_df)

merged_df = pd.merge(movies_df, reddit_df, left_on='Title', right_on='movie_title')
print("Pandas df merged")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x = merged_df[['comment_sentiment']]
y = merged_df['BoxOffice']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("ready to train")
model = RandomForestRegressor()
model.fit(x_train, y_train)
print("Trained, regression fit done")


predictions = model.predict(x_test)
import matplotlib.pyplot as plt


y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel('Actual BoxOffice Earnings')
plt.ylabel('Predicted BoxOffice Earnings')
plt.title('Actual vs Predicted BoxOffice Earnings')
plt.show()
import pickle


with open('first_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
