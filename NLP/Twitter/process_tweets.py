#################### Data Pre-processing: tweets on Twitter ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/11/08
# Last Updated: 2021/11/08
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/Twitter/process_tweets.py
#
#
########## Input Data File(s)
#
#tweets.txt
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python process_tweets.py tweets.txt
#
#
########## Output Data File(s)
#
#processed_tweets.txt
#
#
########## References
#
#Natural Language Processing Specialization
#https://www.coursera.org/specializations/natural-language-processing
#
#Course 1:
#Natural Language Processing with Classification and Vector Spaces
#https://www.coursera.org/learn/classification-vector-spaces-in-nlp?specialization=natural-language-processing
#
####################




########## import Python libraries

import sys

import pandas as pd
import matplotlib.pyplot as plt

import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# process_tweet
def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    #
    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    #
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    #
    return tweets_clean




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #process_tweets.py

arg_tweets_file_name = sys.argv[1]    #tweets.txt



########## load tweets


with open(arg_tweets_file_name, mode='rt', encoding='utf-8') as f:
    txt = f.read()
#
print(type(txt))    #<class 'str'>
#
lst = txt.splitlines()


with open('processed_tweets.txt', 'w') as f:
    for i in range(len(lst)):
        print(lst[i])
        print(process_tweet(lst[i]))
        print(type(process_tweet(lst[i])))    #<class 'list'>
        #
        for d in process_tweet(lst[i]):
            f.write("%s, " % d)
        #
        f.write("\n")


