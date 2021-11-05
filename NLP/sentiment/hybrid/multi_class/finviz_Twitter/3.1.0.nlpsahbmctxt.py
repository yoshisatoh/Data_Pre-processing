#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/11/03
# Last Updated: 2021/11/03
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sentiment/hybrid/multi_class/finviz_Twitter/3.1.0.nlpsahbmctxt.py
#
#
########## Input Data File(s)
#
#
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python 3.1.0.nlpsahbmctxt.py 2.2.2.df_parsed_news.csv
#
#Generally,
#python 3.1.0.nlpsahbmctxt.py (arg_csv_file_name: csv file with 'ticker', 'date', 'time', and 'text' columns)
#
#
########## Output Data File(s)
#
#3.1.1.df_textblob.csv
#3.1.2.df_vs.csv
#3.1.3.df_flr.csv
#3.1.4.df_textblob_vs_flr.csv
#3.1.5.df_parsed_news__df_textblob_vs_flr.csv
#
#
########## References
#
#Sentiment Analysis in Python: TextBlob vs Vader Sentiment vs Flair vs Building It From Scratch
#https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
#
#
#
####################


########## install Python libraries (before running this script)
#
#pip install textblob --upgrade
#python -m pip install textblob
#python -m pip install textblob==0.17.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade textblob --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade vaderSentiment --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
#pip install --upgrade torch --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
#pip install --upgrade tiny-tokenizer --trusted-host pypi.org --trusted-host files.pythonhosted.org
#You'll see the following error;
#Exception: You tried to install "tiny_tokenizer". The name of tiny_tokenizer renamed to "konoha"
#
#pip install --upgrade konoha --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade flair --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## import Python libraries

import sys

import pandas as pd

from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from flair.models import TextClassifier
from flair.data import Sentence


import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import warnings
warnings.simplefilter('ignore')




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #2.3.0.nlpsahbmctxt.py

arg_csv_file_name = str(sys.argv[1])    #2.2.2.df_parsed_news.csv
#
'''
with open(arg_txt_file_name, "r") as tf:
    words_sentences = tf.read().split('\n')
#    
for i in words_sentences:
    print(i)
#
print(words_sentences)
print(type(words_sentences))
'''

df_parsed_news = pd.read_csv(arg_csv_file_name, sep=',', header=0)
print(df_parsed_news.head())




########## 1. Rule-based sentiment analysis

'''
The main drawback with the rule-based approach for sentiment analysis is that the method only cares about individual words and completely ignores the context in which it is used. 

For example, "the party was savage" will be negative when considered by any token-based algorithms.
'''

##### 1.1. Textblob
'''
It is a simple python library that offers API access to different NLP tasks such as sentiment analysis, spelling correction, etc.

Textblob sentiment analyzer returns two properties for a given input sentence: 

Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments. 
Subjectivity is also a float which lies in the range of [0,1]. Subjective sentences with higher number like 1 generally refer to personal opinion, emotion, or judgment. 

Textblob will ignore the words that it doesn't know, it will consider words and phrases that it can assign polarity to and averages to get the final score.
'''

df_textblob = []

for i in range(len(df_parsed_news['text'])):
    #
    textblob   = TextBlob(df_parsed_news['text'][i])
    #
    #print(textblob.sentiment)
    #print(type(textblob.sentiment))
    #
    print(df_parsed_news['text'][i])
    #
    print(textblob.sentiment.polarity)
    print(textblob.sentiment.subjectivity)
    #
    #print(pd.DataFrame(data=[[1, 2, 3]], columns=['A','B','C'], index=None))
    #print(pd.DataFrame(data=[[words_sentences[i], textblob.sentiment.polarity, textblob.sentiment.subjectivity]], columns=['words_sentences','polarity','subjectivity'], index=None))
    if i == 0:
        df_textblob = pd.DataFrame(data=[[df_parsed_news['ticker'][i], df_parsed_news['date'][i], df_parsed_news['time'][i], df_parsed_news['text'][i], textblob.sentiment.polarity, textblob.sentiment.subjectivity]], columns=['ticker','date','time','text','polarity','subjectivity'], index=None)
    else:
        df_textblob = df_textblob.append({'ticker' : df_parsed_news['ticker'][i], 'date' : df_parsed_news['date'][i], 'time' : df_parsed_news['time'][i], 'text' : df_parsed_news['text'][i], 'polarity' : textblob.sentiment.polarity, 'subjectivity' : textblob.sentiment.subjectivity}, ignore_index=True)
    #
#
df_textblob.to_csv('3.1.1.df_textblob.csv', header=True, index=False)




##### 1.2. Vader Sentiment

'''
Valence aware dictionary for sentiment reasoning (VADER) is another popular rule-based sentiment analyzer. 

It uses a list of lexical features (e.g. word) which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment.   

Vader sentiment returns the probability of a given input sentence to be 

positive, negative, and neutral. 

Vader is optimized for social media data and can yield good results when used with data from twitter, facebook, etc.
'''


analyzer = SentimentIntensityAnalyzer()

#vs = analyzer.polarity_scores(str_sentence)
#print("{:-<65} {}".format(str_sentence, str(vs)))    #{'compound': 0.6588, 'neg': 0.0, 'neu': 0.406, 'pos': 0.594}

df_vs = []

for i in range(len(df_parsed_news['text'])):
    #
    vs = analyzer.polarity_scores(df_parsed_news['text'][i])
    #
    print(df_parsed_news['text'][i])
    #
    print(vs)
    print(vs['neg'])
    print(vs['neu'])
    print(vs['pos'])
    print(vs['compound'])
    #
    if i == 0:
        df_vs = pd.DataFrame(data=[[df_parsed_news['ticker'][i], df_parsed_news['date'][i], df_parsed_news['time'][i], df_parsed_news['text'][i], vs['neg'], vs['neu'], vs['pos'], vs['compound']]], columns=['ticker', 'date', 'time', 'text', 'neg', 'neu', 'pos', 'compound'], index=None)
    else:
        df_vs = df_vs.append({'ticker' : df_parsed_news['ticker'][i], 'date' : df_parsed_news['date'][i], 'time' : df_parsed_news['time'][i], 'text' : df_parsed_news['text'][i], 'neg' : vs['neg'], 'neu' : vs['neu'], 'pos' : vs['pos'], 'compound' : vs['compound']}, ignore_index=True)
    #
#
df_vs.to_csv('3.1.2.df_vs.csv', header=True, index=False)




########## 2. Embedding based models

'''
Text embeddings are a form of word representation in NLP in which synonymically similar words are represented using similar vectors which when represented in an n-dimensional space will be close to each other.

'''

##### 2.1. Flair 


classifier = TextClassifier.load('en-sentiment')

'''
str_sentence = 'The food was great!'
sentence = Sentence(str_sentence)
classifier.predict(sentence)
print('Sentence above is: ', sentence.labels)    #[POSITIVE (0.9961)
print(sentence.labels)    #[POSITIVE (0.9961)
exit()
'''



df_flr = []

for i in range(len(df_parsed_news['text'])):
    #
    print(df_parsed_news['text'][i])    #Tesla possibly goes up.
    #
    #print(Sentence(df_parsed_news['text'][i]))    #Sentence: "Tesla possibly goes up ."   [− Tokens: 5]
    #
    df_parsed_news_text = Sentence(df_parsed_news['text'][i])
    #
    classifier.predict(df_parsed_news_text)
    #
    #print(type(df_parsed_news_text.labels))    #<class 'list'>
    print(df_parsed_news_text.labels)    #[POSITIVE (0.956)]
    print(str(str(df_parsed_news_text.labels).split(' (')[0]).split('[')[1])    #POSITIVE
    print(str(str(df_parsed_news_text.labels).split(' (')[1]).split(')')[0])    #0.956
    #
    if i == 0:
        df_flr = pd.DataFrame(data=[[df_parsed_news['ticker'][i], df_parsed_news['date'][i], df_parsed_news['time'][i], df_parsed_news['text'][i], str(str(df_parsed_news_text.labels).split(' (')[0]).split('[')[1], str(str(df_parsed_news_text.labels).split(' (')[1]).split(')')[0]]], columns=['ticker', 'date', 'time', 'text', 'pos_neg', 'value'], index=None)
    else:
        df_flr = df_flr.append({'ticker' : df_parsed_news['ticker'][i], 'date' : df_parsed_news['date'][i], 'time' : df_parsed_news['time'][i], 'text' : df_parsed_news['text'][i], 'pos_neg' : str(str(df_parsed_news_text.labels).split(' (')[0]).split('[')[1], 'value' : str(str(df_parsed_news_text.labels).split(' (')[1]).split(')')[0]}, ignore_index=True)
    #
#
df_flr.to_csv('3.1.3.df_flr.csv', header=True, index=False)




########## 3. Merging all the output files and then output the result

df_textblob_vs     = pd.merge(df_textblob, df_vs, on=['ticker', 'date', 'time', 'text'], how='inner')
df_textblob_vs_flr = pd.merge(df_textblob_vs, df_flr, on=['ticker', 'date', 'time', 'text'], how='inner')

df_textblob_vs_flr.to_csv('3.1.4.df_textblob_vs_flr.csv', header=True, index=False)




df_parsed_news__df_textblob_vs_flr = pd.merge(df_parsed_news, df_textblob_vs_flr, on=['ticker', 'date', 'time', 'text'], how='outer')
df_parsed_news__df_textblob_vs_flr.to_csv('3.1.5.df_parsed_news__df_textblob_vs_flr.csv', header=True, index=False)
