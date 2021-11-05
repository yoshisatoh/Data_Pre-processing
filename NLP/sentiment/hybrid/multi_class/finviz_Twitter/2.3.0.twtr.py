#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis (Embedding-based, Binary Class, Data on Twitter) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/11/05
# Last Updated: 2021/11/05
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/2.3.0.twtr.py
#
#
########## Input Data File(s)
#
#2.3.0.bearer_token.txt    #BEARER_TOKEN: Your Bearer Token on Twitter
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python 2.3.0.twtr.py tesla 10 TSLA
#
#Generally,
#python 2.3.0.twtr.py (SEARCH_QUERY) (SEARCH_COUNT) (arg_ticker)
#
#
########## Output Data File(s)
#
#2.3.1.tweets.csv
#2.3.2.tweets.csv
#2.3.3.tweets.csv
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

import re



########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlpsarbmc.py

SEARCH_QUERY = str(sys.argv[1])    #'tesla'
SEARCH_COUNT = int(sys.argv[2])    #100
arg_ticker   = str(sys.argv[3])    #TSLA

#BEARER_TOKEN
with open('2.3.0.bearer_token.txt') as fp:
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
df.to_csv('2.3.1.tweets.csv', header=True, index=False)

#exit()



########## 4. Tweet Pre-processing


##### Analyzing Tweets
#Most of our tweets are very messy. Cleaning text data is fundamental, although we will just do the bare minimum in this example.
whitespace  = re.compile(r"\s+")
web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
searchq     = re.compile(r"(?i)@" + SEARCH_QUERY + "(?=\b)")
user        = re.compile(r"(?i)@[a-z0-9_]+")


##### Have our clean(ish) tweets

df['ticker'] = arg_ticker
df['date']   = ""
df['time']   = ""

#for tweet in tweets['text'].to_list():
#for tweet in tweets.json()['full_text']:
#for twt in tweets.json()['statuses']:
#
#for tweet in df['text']:
for i in range(len(df['text'])):
    #
    tweet = df['text'][i]
    #
    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = searchq.sub(SEARCH_QUERY, tweet)
    tweet = user.sub('', tweet)
    #
    df['text'][i] = tweet
    #
    #
    df['date'][i] = datetime.strptime(df['created_at'][i][4:10] + ' ' + df['created_at'][i][26:30], "%b %d %Y")
    #print(df['date'][i])
    #print(type(df['date'][i]))
    df['date'][i] = str(df['date'][i])[0:10]
    #
    print(str(df['created_at'][i]))
    df['time'][i] = str(df['created_at'][i])[11:19]


#print(df['created_at'][0])
#print(type(df['created_at'][0]))    #<class 'str'>
#print(df['created_at'][0][4:19])
#
#print(df['created_at'][0][4:10] + ' ' + df['created_at'][0][26:30])    #Nov 05 2021
'''
print(
    datetime.strptime(
        df['created_at'][0][4:10] + ' ' + df['created_at'][0][26:30],
        "%b %d %Y"
    )
)    #2021-11-05 00:00:00
'''
#
print(df['created_at'][0][11:19])    #12:18:40

df.to_csv('2.3.2.tweets.csv', header=True, index=False)



df1 = pd.read_csv('2.3.2.tweets.csv', header=0, usecols=['ticker', 'date', 'time', 'text'])
df2 = df1[['ticker', 'date', 'time']]
df3 = pd.concat([df2, df1['text']], axis=1)
df3.to_csv('2.3.3.tweets.csv', header=True, index=False)