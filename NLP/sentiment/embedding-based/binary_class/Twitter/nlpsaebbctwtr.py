#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis (Embedding-based, Binary Class, Data on Twitter) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/27
# Last Updated: 2021/10/28
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sentiment/embedding-based/binary_class/Twitter/nlpsaebbctwtr.py
#
#
########## Input Data File(s)
#
#bearer_token.txt    #BEARER_TOKEN: Your Bearer Token on Twitter
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlpsaebbctwtr.py tesla 100 TSLA
#
#Generally,
#python nlpsaebbctwtr.py (SEARCH_QUERY) (SEARCH_COUNT) (arg_ticker: a ticker on Yahoo! Finance)
#
#
########## Output Data File(s)
#
#tweets.csv
#ticker_history.csv
#
#
#
#
########## References
#
#Sentiment Analysis for Stock Price Prediction in Python
#How we can predict stock price movements using Twitter
#https://towardsdatascience.com/sentiment-analysis-for-stock-price-prediction-in-python-bed40c65d178
#
#
#
####################




########## apply for a developer account on Twitter
'''
In order to access Twitter APIs, you must first apply for a developer account.
https://developer.twitter.com/
Apply
Apply for a developer account
'''




########## install Python libraries (before running this script)
#
#pip install textblob --upgrade
#pip install textblob --U
#python -m pip install textblob
#python -m pip install textblob==0.17.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade textblob --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade vaderSentiment --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade flair --trusted-host pypi.org --trusted-host files.pythonhosted.org


#If you see an error:
#TypeError: Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type
#
#then use numpy 1.18.1 to avoid incompatibility with old pandas
#pip install --upgrade numpy==1.18.1




########## import Python libraries

import sys

from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

import requests

import flair

import yfinance as yf

import re



########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlpsarbmc.py

SEARCH_QUERY = str(sys.argv[1])    #'tesla'
SEARCH_COUNT = int(sys.argv[2])    #100
arg_ticker   = str(sys.argv[3])    #'TSLA'

#BEARER_TOKEN
with open('bearer_token.txt') as fp:
    BEARER_TOKEN = fp.read()




########## 0. Embedding based models

'''
Text embeddings are a form of word representation in NLP in which synonymically similar words are represented using similar vectors which when represented in an n-dimensional space will be close to each other.

'''




########## 1. Import Libraries

# See the "import Python libraries" section above.




########## 2. Twitter API

#To search recent tweets
#https://api.twitter.com/1.1/tweets/search/recent

#https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
params = {'q': SEARCH_QUERY,
          'tweet_mode': 'extended',
          'lang': 'en',
          'count': str(SEARCH_COUNT)
}
#response = requests.get(
tweets = requests.get(
    'https://api.twitter.com/1.1/search/tweets.json',
    params=params,
    headers={'authorization': 'Bearer ' + BEARER_TOKEN}
)




########## 3. Building Our Dataset

def get_data(tweet):
    data = {
        'id': tweet['id_str'],
        'created_at': tweet['created_at'],
        'text': tweet['full_text']
    }
    return data


df = pd.DataFrame()

#for twt in response.json()['statuses']:
for twt in tweets.json()['statuses']:
    row = get_data(twt)
    df = df.append(row, ignore_index=True)


print(df.head())
df.to_csv('df.csv', header=True, index=True)

#exit()



########## 4. Sentiment Analysis


##### Flair

sentiment_model = flair.models.TextClassifier.load('en-sentiment')

#tokenize our text
#sentence = flair.data.Sentence(TEXT)
#sentence = flair.data.Sentence(<TEXT HERE>)

#add the sentiment rating to the data stored in sentence
#sentiment_model.predict(sentence)


##### Analyzing Tweets
#Most of our tweets are very messy. Cleaning text data is fundamental, although we will just do the bare minimum in this example.
whitespace  = re.compile(r"\s+")
web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
searchq     = re.compile(r"(?i)@" + SEARCH_QUERY + "(?=\b)")
user        = re.compile(r"(?i)@[a-z0-9_]+")


##### Have our clean(ish) tweets

#print(type(SEARCH_QUERY))
#print(type(tweet))
#print(tweet)

# we then use the sub method to replace anything matching
#tweet = whitespace.sub(' ', tweet)
#tweet = web_address.sub('', tweet)
#tweet = searchq.sub(SEARCH_QUERY, tweet)
#tweet = user.sub('', tweet)


##### Tokenize it by converting it into a sentence object, and then predict the sentiment:
#sentence = flair.data.Sentence(tweet)
#sentiment_model.predict(sentence)


##### extract our predictions and add them to our tweets dataframe
#probability = sentence.labels[0].score  # numerical value 0-1
#sentiment = sentence.labels[0].value  # 'POSITIVE' or 'NEGATIVE'


# we will append probability and sentiment preds later
probs = []
sentiments = []

# use regex expressions (in clean function) to clean tweets
#tweets['text'] = tweets['text'].apply(clean)

#print(type(tweets))
#print(tweets['text'])


#for tweet in tweets['text'].to_list():
#for tweet in tweets.json()['full_text']:
#for twt in tweets.json()['statuses']:
for tweet in df['text']:
    #
    #tweet = get_data(twt)['text']
    #
    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = searchq.sub(SEARCH_QUERY, tweet)
    tweet = user.sub('', tweet)
    #
    # make prediction
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)
    # extract sentiment prediction
    probs.append(sentence.labels[0].score)  # numerical score 0-1
    sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

# add probability and sentiment predictions to tweets dataframe
#tweets['probability'] = probs
#tweets['sentiment'] = sentiments


#print(tweets.head())
#tweets.to_csv('tweets.csv', header=True, index=True)

print(type(probs))    #<class 'list'>
print(len(probs))     #100
df_probs = pd.DataFrame(probs, columns=['probs'])
print(df_probs.head())
#df_probs.to_csv('df_probs.csv', header=True, index=True)


#sentiments.to_csv('sentiments.csv', header=True, index=True)
df_sentiments = pd.DataFrame(sentiments, columns=['sentiments'])
print(df_sentiments.head())
#df_sentiments.to_csv('df_sentiments.csv', header=True, index=True)


df_df_probs = pd.merge(df, df_probs, left_index=True, right_index=True)
df_df_probs_df_sentiments = pd.merge(df_df_probs, df_sentiments, left_index=True, right_index=True)
df_df_probs_df_sentiments.to_csv('df_df_probs_df_sentiments.csv', header=True, index=True)




########## Yahoo! Finance

ticker = yf.Ticker(arg_ticker)

arg_start    = '2020-10-28'
arg_end      = '2021-10-27'
arg_interval = '1d'    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')

ticker_history = ticker.history(
    start    = arg_start,
    end      = arg_end,
    interval = arg_interval
).reset_index()

'''
ticker_history = ticker.history(
    interval = '60m'
).reset_index()
'''
#print(ticker_history.head())
'''
                   Datetime        Open        High         Low       Close   Volume  Dividends  Stock Splits
0 2021-09-27 09:30:00-04:00  773.119995  779.769897  769.309998  776.830017  7074175          0             0
1 2021-09-27 10:30:00-04:00  776.890015  789.599976  776.250000  788.895996  5437870          0             0
2 2021-09-27 11:30:00-04:00  788.897278  794.880005  788.090027  794.429871  3693661          0             0
3 2021-09-27 12:30:00-04:00  794.341492  797.000000  791.500122  793.089294  3071720          0             0
4 2021-09-27 13:30:00-04:00  793.090027  797.489990  791.299988  795.643982  2294079          0             0
'''
ticker_history.to_csv('ticker_history.csv', header=True, index=True)