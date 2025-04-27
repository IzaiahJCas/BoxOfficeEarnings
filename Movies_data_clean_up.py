import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, when, length, trim, udf, col, avg, count as spark_count, round, sum as spark_sum
from pyspark.sql.types import StringType, FloatType, IntegerType, DoubleType
import pandas as pd
from pyspark.ml.feature import Imputer
from datetime import datetime
import os
import glob
import shutil
from supabase import create_client
from dotenv import load_dotenv
import numpy as np

df = pd.read_csv('Movies_rows.csv')
numeric_cols = [
    'imdbID', 'Year' , 'Released','Runtime', 'Metascore','imdbRating'
    ,'imdbVotes','BoxOffice', 'Internet Movie Database', 'Rotten Tomatoes','Metacritic'
]
text_cols = [
    'imdbID', 'Title', 'Rated', 'Genre', 'Director',
    'Writer', 'Actors', 'Plot', 'Language', 'Country', 'Awards'
]
df[numeric_cols].to_csv('Movies_numeric.csv', index=False)
df[text_cols].to_csv('Movies_text.csv', index=False)

# Initialize Spark session and read CSV
spark = SparkSession.builder.appName("MoviesCleaning").getOrCreate()
df = spark.read \
    .option("header", True) \
    .option("quote", '"') \
    .option("escape", "\\") \
    .option("escape","\"")  \
    .option("multiline", True) \
    .option("ignoreLeadingWhiteSpace", True) \
    .option("ignoreTrailingWhiteSpace", True) \
    .option("mode", "DROPMALFORMED") \
    .option("delimiter", ",") \
    .csv("Movies_numeric.csv")


# Change the data type to datetime
def convert_date(date_str):
    if not date_str or str(date_str).strip() == '':
        return None
    s = str(date_str).strip()
    if len(s) == 10 and s[4] == '-' and s[7] == '-':
        return s
    try:
        return datetime.strptime(s, '%d %b %Y').strftime('%Y-%m-%d')
    except:
        return None

convert_date_udf = udf(convert_date, StringType())
df = df.withColumn('Released', convert_date_udf(col('Released')))

# Remove min from 'Runtime' and convert to int
df = df.withColumn(
    'Runtime',
    regexp_replace('Runtime', ' min', '').cast(IntegerType())
)

# Change data type to float
df = df.withColumn(
    'Metascore',
    col('Metascore').cast(FloatType())
)

# Change data type to int
df = df.withColumn(
    'imdbVotes',
    regexp_replace('imdbVotes', ',', '').cast(IntegerType())
)

# Change data type to float and drop blanks
df = df.withColumn(
    'BoxOffice',
    regexp_replace('BoxOffice', '[$,]', '').cast('long')
)
df = df.filter(col('BoxOffice').isNotNull())

# Remove '%', convert to float
df = df.withColumn(
    'Rotten Tomatoes',
    regexp_replace('Rotten Tomatoes', '%', '').cast(IntegerType())
)

# Change Metacritic to float
df = df.withColumn(
    'Metacritic',
    col('Metacritic').cast(IntegerType())
)

df = df.withColumn('imdbRating', round(col('imdbRating'), 1))
df = df.withColumn('Internet Movie Database', round(col('Internet Movie Database'), 1))

# Your list of columns to process
columns = ['Metascore', 'imdbRating', 'imdbVotes', 'Internet Movie Database', 'Rotten Tomatoes', 'Metacritic']

row_count = df.count()
cols_to_drop = []
cols_to_impute = []

for c in columns:
    # Count nulls or blanks
    blank_count = df.filter((col(c).isNull()) | (col(c) == '')).count()
    blank_percent = blank_count / row_count

    print(f"{c}: {blank_count} blanks ({blank_percent*100:.2f}%)")
    if blank_count == 0:
        print(f"  {c} has no missing values. Skipping imputation.")
        continue
    if blank_percent < 0.3:
        # Mean imputation (if numeric)
        print(f"  {c} eligible for mean imputation.")
        cols_to_impute.append(c)                # or "median"
    else:
        print(f"  Dropping column {c} due to too many blanks.")
        cols_to_drop.append(c)

if cols_to_drop:
    df = df.drop(*cols_to_drop)

if cols_to_impute:
    # Impute mean for all eligible columns at once
    imputer = Imputer(
        inputCols=cols_to_impute,
        outputCols=cols_to_impute,  # in-place
        strategy="mean"
    )
    model = imputer.fit(df)
    df = model.transform(df)
    print(f"Imputed mean for columns: {cols_to_impute}")

df = df.withColumn('Metascore', round(col('Metascore'), 1))
# Write cleaned data to a new CSV file
df.write.mode('overwrite').option("header", True).csv("Movies_rows_cleaned_spark")

spark_csv = glob.glob('Movies_rows_cleaned_spark/part-*.csv')[0]
df_numeric = pd.read_csv(spark_csv)
df_text = pd.read_csv('Movies_text.csv')
merged = pd.merge(df_numeric, df_text, on='imdbID', how='left')
column_order = [
    'imdbID', 'Title', 'Year','Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer',
    'Actors', 'Plot', 'Language', 'Country', 'Awards', 'Metascore', 'imdbRating',
    'imdbVotes', 'BoxOffice', 'Internet Movie Database', 'Rotten Tomatoes', 'Metacritic'
]
merged = merged[column_order]
merged.to_csv('Movies_final_combined.csv', index=False)
os.remove('Movies_text.csv')
shutil.rmtree('Movies_rows_cleaned_spark')
os.remove('Movies_numeric.csv')

df = pd.read_csv('Movies_final_combined.csv')
#@param your actual .env file or make it blank
load_dotenv('.env.local')
url = os.getenv("supabaseURL")
key = os.getenv("key")
supabase = create_client(url, key)

df = df.replace([np.nan, np.inf, -np.inf], None)
data = df.to_dict(orient='records')
batch_size = 500
print(len(data))
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    response = supabase.table('MoviesCleaned').insert(batch).execute()