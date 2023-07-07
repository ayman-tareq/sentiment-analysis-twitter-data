
## data cleaning

import pandas as pd
import json, os, re
import numpy as np
import pycountry

def find_country_from_place_column(x):
  if pd.isna(x):
    return np.nan
  else:
    try:
      country = re.findall(r"\'country\':\s\'(.*)[\'\"],", x)[0]
      return country
    except Exception as e:
      return np.nan



df = pd.read_csv('1.2m_combined_dataset.csv', encoding='utf-8')

# df.to_excel('tem.xlsx', index=False)

len(df)-df['place'].isnull().sum()

df.drop_duplicates(['tweet_content'], inplace=True)
# df.duplicated(['tweet_content']).sum()
df = df[['tweet_url', 'date', 'user_name', 'tweet_content', 'lang', 'is_user_verified', 'user_location', 'place']]

df['place'].isnull().sum()

df = df.fillna(np.nan)

os.system('cls')

df['place_to_country'] = df['place'].apply(find_country_from_place_column)

df['place_to_country'].notnull().sum()
df['place_to_country'].sample(10)


df['user_location'].notnull().sum()
df['user_location'].isnull().sum()

df['user_location'] = df['user_location'].fillna(df['place_to_country']).where(df['place_to_country'].notnull(), df['user_location'])

df['user_location'].notnull().sum()
df['user_location'].isnull().sum()

df.drop(['place_to_country', 'place'], axis=1, inplace=True)

df = df.fillna(method='ffill')

df.to_csv('1_1.2m_cleaned_data.csv', index=False, encoding='utf-8')

# most_frequent_value = df['user_location'].value_counts().idxmax()
# df['user_location'].fillna(most_frequent_value, inplace=True)

# df2 = df.fillna(method='bfill')

#read a csv file
df1 = pd.read_csv('1_1.2m_cleaned_data.csv', encoding='utf-8')
df1.shape
df1['user_location'].notnull().sum()
df1['user_name'].nunique()



# df1['user_location'].value_counts().idxmax()

#----------------------- Analyze the data -----------------------#

# Import Libraries

from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import re
import string
import seaborn as sns


from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


# split a pandas df into chunk
def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

df = pd.read_csv('1m_clead_data_with_country.csv', encoding='utf-8')


# Create a dictionary mapping country names to alpha-2 codes
country_dict = {country.name: country.alpha_2 for country in pycountry.countries}

# Convert country names in the DataFrame to alpha-2 codes
df['alpha_2_code'] = df['country'].map(country_dict)

df.to_csv('1m_clean_data_with_country_and_code.csv', index=False, encoding='utf-8')

df = pd.read_csv('1m_clean_data_with_country_and_code.csv', encoding='utf-8')

df['tweet_url'].isnull().sum()

# slicing the date , and removing the time portion
df['date'] = df.date.str.slice(0, 10)

# checking all the unique dates in the dataset
df['date'].unique()
len(df['date'].unique())

# checking how many unique language
# df are present in the dataset
print(df["lang"].unique())

# Removing RT, Punctuation etc
def remove_rt(x): 
  return re.sub('RT @\w+: ', " ", x)

def rt(x): 
  return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)

df["content"] = df.tweet_content.map(remove_rt).map(rt)
df["content"] = df.content.str.lower()

df.iloc[10000,3]
df.iloc[10000,-1]


df[['polarity', 'subjectivity']] = df['content'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
  
for index, row in df.iterrows():
  # break
  print(index)
  content = row['content']
  score = SentimentIntensityAnalyzer().polarity_scores(content)
  neg = score['neg']
  neu = score['neu']
  pos = score['pos']
  comp = score['compound']
	
  if neg > pos:
    df.loc[index, 'sentiment'] = "negative"
  elif pos > neg:
    df.loc[index, 'sentiment'] = "positive"
  else:
    df.loc[index, 'sentiment'] = "neutral"
		
  df.loc[index, 'neg'] = neg
  df.loc[index, 'neu'] = neu
  df.loc[index, 'pos'] = pos
  df.loc[index, 'compound'] = comp


df.shape
list(df)
df
df.to_csv('1m_with_sentiment_score.csv', index=False, encoding='utf-8')

df = pd.read_csv('1m_with_sentiment_score.csv', encoding='utf-8')

df[["content", "sentiment", "polarity",
		"subjectivity", "neg", "neu", "pos"]].head(5)


# Removing Punctuation
def remove_punct(text):
	text = "".join([char for char in text if
					char not in string.punctuation])
	text = re.sub('[0-9]+', '', text)
	return text


df['punct'] = df['content'].apply(lambda x: remove_punct(x))

# Applying tokenization
def tokenization(text):
	text = re.split('\W+', text)
	return text


df['tokenized'] = df['punct'].apply(lambda x: tokenization(x.lower()))

# Removing stopwords
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
	text = [word for word in text if
			word not in stopword]
	return text

df['nonstop'] = df['tokenized'].apply(lambda x: remove_stopwords(x))

# Applying Stemmer
ps = nltk.PorterStemmer()

def stemming(text):
	text = [ps.stem(word) for word in text]
	return text

df['stemmed'] = df['nonstop'].apply(lambda x: stemming(x))

df.head()

df.to_csv('data_with_tokenized_20cols.csv', index=False, encoding='utf-8')


##################


df = pd.read_csv('data_with_tokenized_20cols.csv', encoding='utf-8')

grouped = df.groupby('country')

tot_tweets = grouped.size() # country, count
verified_counts = grouped['is_user_verified'].sum()
user_counts = grouped['user_name'].nunique()


data = {'total users': user_counts, 
        'verified users': verified_counts, 
        'total tweets':tot_tweets}

df1 = pd.DataFrame(data)

# Displaying the DataFrame
df1.sort_values(['total tweets'], inplace=True, ascending=False)
df1.to_excel('country_users.xlsx')

df['year'] = pd.to_datetime(df['date']).dt.year

year_counts = df.groupby('year').size()
verified_counts_by_year = df[df['is_user_verified'] == True].groupby('year').size()
user_counts_by_year = df.groupby('year')['user_name'].nunique()

data = {'total users': user_counts_by_year, 
        'verified users': verified_counts_by_year, 
        'total tweets':year_counts}

df1 = pd.DataFrame(data)
df1.to_excel('year_counts.xlsx')

# keep top 150 countries that have maximum tweets
df_top_countries = df[df['country'].isin(list(df1[:150].index))]

df_top_countries.groupby(['country']).size()

sub_df1 = df_top_countries[['country', 'polarity', 'subjectivity', 'neg', 'neu', 'pos', 'compound']]


country_sentiment = sub_df1.groupby('country').mean()[['polarity', 'subjectivity', 'neg', 'neu', 'pos', 'compound']]
country_sentiment_sorted = country_sentiment.sort_values(by='compound', ascending=False)

country_sentiment_sorted.to_excel('country_sentiment_sorted.xlsx')



# Define sentiment categories based on compound score
country_sentiment_sorted['sentiment'] = country_sentiment_sorted['compound'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Calculate the count and percentage of countries in each sentiment category
sentiment_counts = country_sentiment_sorted['sentiment'].value_counts()
sentiment_percentages = sentiment_counts / len(country_sentiment_sorted) * 100

# Create a pie chart
labels = sentiment_percentages.index
sizes = sentiment_percentages.values
colors = ['lightgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1)  # Explode the positive sentiment slice

plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
plt.title('Sentiment towards the war')

plt.show()


df = df_top_countries

tem_df = df_top_countries

total_pos = len(tem_df.loc[tem_df['sentiment'] == "positive"])
total_neg = len(tem_df.loc[tem_df['sentiment'] == "negative"])
total_neu = len(tem_df.loc[tem_df['sentiment'] == "neutral"])

total_tweets = len(tem_df)

print("Total Positive Tweets {} % : {:.2f}".format(total_pos, (total_pos/total_tweets)*100))
print("Total Negative Tweets {} % : {:.2f}".format(total_neg, (total_neg/total_tweets)*100))
print("Total Neutral Tweets {} % : {:.2f}".format(total_neu, (total_neu/total_tweets)*100))


mylabels = ["Positive", "Negative", "Neutral"]
mycolors = ['lightgreen', 'lightcoral', 'lightskyblue']
# mycolors = ["Green", "Red", "Blue"]

# plt.figure(figsize=(8, 5),
# 		dpi=600) # Push new figure on stack
myexplode = [0, 0.1, 0]
plt.pie([total_pos, total_neg, total_neu], colors=mycolors,
		labels=mylabels, explode=myexplode, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
# plt.title('sentiment towards the war', loc='center', y=1)

plt.show()


df1 = df[df['date'] < '2022-02-01']  # Filter records before 1st Feb, 2022
df2 = df[df['date'] >= '2022-02-01']  # Filter records from 1st Feb, 2022 onwards

# df2.groupby('country').size()

tweets = df1

pos_list = []
neg_list = []
neu_list = []
for i in tweets["date"].unique():
	temp = tweets[tweets["date"] == i]
	positive_temp = temp[temp["sentiment"] == "positive"]
	negative_temp = temp[temp["sentiment"] == "negative"]
	neutral_temp = temp[temp["sentiment"] == "neutral"]
	pos_list.append(((positive_temp.shape[0]/temp.shape[0])*100, i))
	neg_list.append(((negative_temp.shape[0]/temp.shape[0])*100, i))
	neu_list.append(((neutral_temp.shape[0]/temp.shape[0])*100, i))

neu_list = sorted(neu_list, key=lambda x: x[1])
pos_list = sorted(pos_list, key=lambda x: x[1])
neg_list = sorted(neg_list, key=lambda x: x[1])

x_cord_neg = []
y_cord_neg = []

x_cord_pos = []
y_cord_pos = []

x_cord_neu = []
y_cord_neu = []

for i in neg_list:
	x_cord_neg.append(i[0])
	y_cord_neg.append(i[1])


for i in pos_list:
	x_cord_pos.append(i[0])
	y_cord_pos.append(i[1])

for i in neu_list:
	x_cord_neu.append(i[0])
	y_cord_neu.append(i[1])


# plt.figure(figsize=(16, 9), dpi=100) 

plt.plot(y_cord_neg, x_cord_neg, label="negative",
		color="red")
plt.plot(y_cord_pos, x_cord_pos, label="positive",
		color="green")
plt.plot(y_cord_neu, x_cord_neu, label="neutral",
		color="blue")

plt.xticks(np.arange(0, len(tweets["date"].unique()) + 1, 25))

plt.xticks(rotation=90)
plt.grid(axis='y')

plt.legend()
plt.show()