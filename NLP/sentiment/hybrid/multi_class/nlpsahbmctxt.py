#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/27
# Last Updated: 2021/10/27
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sentiment/hybrid/multi_class/nlpsahbmctxt.py
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
#python nlpsahbmctxt.py words_sentences.txt
#
#Generally,
#python nlpsahbmctxt.py (text file)
#
#
########## Output Data File(s)
#
#df_textblob.csv
#df_vs.csv
#df_flr.csv
#df_textblob_vs_flr.csv
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

#print(sys.argv[0])    #nlpssentiment.py

arg_txt_file_name = str(sys.argv[1])    #words_sentences.txt
#
with open(arg_txt_file_name, "r") as tf:
    words_sentences = tf.read().split('\n')
#    
for i in words_sentences:
    print(i)
#
print(words_sentences)
print(type(words_sentences))




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

for i in range(len(words_sentences)):
    #
    textblob   = TextBlob(words_sentences[i])
    #
    #print(textblob.sentiment)
    #print(type(textblob.sentiment))
    #
    print(words_sentences[i])
    #
    print(textblob.sentiment.polarity)
    print(textblob.sentiment.subjectivity)
    #
    #print(pd.DataFrame(data=[[1, 2, 3]], columns=['A','B','C'], index=None))
    #print(pd.DataFrame(data=[[words_sentences[i], textblob.sentiment.polarity, textblob.sentiment.subjectivity]], columns=['words_sentences','polarity','subjectivity'], index=None))
    if i == 0:
        df_textblob = pd.DataFrame(data=[[words_sentences[i], textblob.sentiment.polarity, textblob.sentiment.subjectivity]], columns=['words_sentences','polarity','subjectivity'], index=None)
    else:
        df_textblob = df_textblob.append({'words_sentences' : words_sentences[i], 'polarity' : textblob.sentiment.polarity, 'subjectivity' : textblob.sentiment.subjectivity}, ignore_index=True)
    #
#
df_textblob.to_csv('df_textblob.csv', header=True, index=False)




##### 1.2. VADER sentiment

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

for i in range(len(words_sentences)):
    #
    vs = analyzer.polarity_scores(words_sentences[i])
    #
    print(words_sentences[i])
    #
    print(vs)
    print(vs['neg'])
    print(vs['neu'])
    print(vs['pos'])
    print(vs['compound'])
    #
    if i == 0:
        df_vs = pd.DataFrame(data=[[words_sentences[i], vs['neg'], vs['neu'], vs['pos'], vs['compound']]], columns=['words_sentences','neg', 'neu','pos', 'compound'], index=None)
    else:
        df_vs = df_vs.append({'words_sentences' : words_sentences[i], 'neg' : vs['neg'], 'neu' : vs['neu'], 'pos' : vs['pos'], 'compound' : vs['compound']}, ignore_index=True)
    #
#
df_vs.to_csv('df_vs.csv', header=True, index=False)




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

for i in range(len(words_sentences)):
    #
    print(words_sentences[i])    #Tesla possibly goes up.
    #
    #print(Sentence(words_sentences[i]))    #Sentence: "Tesla possibly goes up ."   [− Tokens: 5]
    #
    word_sentence = Sentence(words_sentences[i])
    #
    classifier.predict(word_sentence)
    #
    #print(type(word_sentence.labels))    #<class 'list'>
    print(word_sentence.labels)    #[POSITIVE (0.956)]
    print(str(str(word_sentence.labels).split(' (')[0]).split('[')[1])    #POSITIVE
    print(str(str(word_sentence.labels).split(' (')[1]).split(')')[0])    #0.956
    #
    if i == 0:
        df_flr = pd.DataFrame(data=[[words_sentences[i], str(str(word_sentence.labels).split(' (')[0]).split('[')[1], str(str(word_sentence.labels).split(' (')[1]).split(')')[0]]], columns=['words_sentences','pos_neg', 'value'], index=None)
    else:
        df_flr = df_flr.append({'words_sentences' : words_sentences[i], 'pos_neg' : str(str(word_sentence.labels).split(' (')[0]).split('[')[1], 'value' : str(str(word_sentence.labels).split(' (')[1]).split(')')[0]}, ignore_index=True)
    #
#
df_flr.to_csv('df_flr.csv', header=True, index=False)




########## 3. Merging all the output files and then output the result

df_textblob_vs     = pd.merge(df_textblob, df_vs, on='words_sentences', how='outer')
df_textblob_vs_flr = pd.merge(df_textblob_vs, df_flr, on='words_sentences', how='outer')

df_textblob_vs_flr.to_csv('df_textblob_vs_flr.csv', header=True, index=False)

