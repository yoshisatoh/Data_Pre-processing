#################### Data Pre-processing: Natural language processing (NLP) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/22
# Last Updated: 2021/10/22
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/web-scraping_sentiment/nlpwebsentiment.py
#
#
########## Input Data File(s)
#
#loughran_mcdonald_master_dic_2016.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlpwebsentiment.py "https://en.wikipedia.org/wiki/SpaceX" 20
#
#Generally,
#python nlpwebsentiment.py (url) (num_words: number of words to show in descending order of counts for each word - if it's 20, then it shows top 20 words)
#
#
########## Output Data File(s)
#
#df.csv
#df_sorted.csv
#df_sentiment.csv
#df_sorted_sentiment.csv
#
#
########## References
#
#Gentle Start to Natural Language Processing using Python
#https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3
#
#Natural Language Toolkit (NLTK)
#https://www.nltk.org/
#
#
####################




########## install Python libraries (before running this script)
#
#pip install nltk --upgrade
#pip install nltk --U
#python -m pip install nltk
#python -m pip install nltk==3.5
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade nltk --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade bs4 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade html5lib --trusted-host pypi.org --trusted-host files.pythonhosted.org


########## import Python libraries

import sys

import nltk
#nltk.download()
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import wordnet
from nltk.corpus import stopwords


import urllib.request
from bs4 import BeautifulSoup
import html5lib

import matplotlib.pyplot as plt

import csv

import pandas as pd


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

#print(sys.argv[0])    #nlp_1_prep.py

url       = str(sys.argv[1])    #'https://en.wikipedia.org/wiki/SpaceX'

num_words = int(sys.argv[2])    #20




########## web page settings

response =  urllib.request.urlopen(url)
html = response.read()
#print(html)




########## getting text from the web page

soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)
#print(text)




########## tokens

tokens = [t for t in text.split()]

#print(tokens)
print(type(tokens))
print(len(tokens))




########## lemmatize_words

def lemmatize_words(words):
    """
    Lemmatize words 
    Parameters
    ----------
    words : list of str
        List of words
    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """
    #
    # TODO: Implement
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    #
    return lemmatized_words



########## word frequency

sr= stopwords.words('english')

clean_tokens = tokens[:]


for token in tokens:
    if token in stopwords.words('english'):
        #
        clean_tokens.remove(token)
        #

freq = nltk.FreqDist(clean_tokens)
#print(freq)
#print(type(freq))
#<class 'nltk.probability.FreqDist'>


'''
for key,val in freq.items():
    #
    print(str(key) + ':' + str(val))
    #
'''


f = open('df.csv' , 'w', encoding="utf-8")
writer = csv.writer(f, lineterminator='\n')
#
for key,val in freq.items():
    writer.writerow([str(key), str(val)])
#
#
f.close()
print(f.closed)



df = pd.read_csv('df.csv', sep=',', header=None, index_col=False)
df = df.rename(columns={0: "word", 1: "count"})
print(df.head())
#print(df.values)


#lemmatize
df['word'] = lemmatize_words(df['word'].str.lower())
df         = df.drop_duplicates('word')


df_sorted = df.sort_values(by=['count'], ascending=False)
print(df_sorted.head())

df_sorted.to_csv('df_sorted.csv', index=False)  




########## word frequency: draw graphs


##### individual counts
fig = plt.figure(1, figsize = (10,8))
plt.gcf().subplots_adjust(bottom=0.20)    # to avoid x-ticks cut-off
freq.plot(num_words, cumulative=False, title='Top ' + str(num_words) + ' Most Common Words in Corpus: Individual')
plt.show()
fig.savefig('Fig_freq_individual.png', bbox_inches = "tight")
plt.close()


##### cumulative counts

fig = plt.figure(2, figsize = (10,8))
plt.gcf().subplots_adjust(bottom=0.20)    # to avoid x-ticks cut-off
freq.plot(num_words, cumulative=True, title='Top ' + str(num_words) + ' Most Common Words in Corpus: Cumulative')
plt.show()
fig.savefig('Fig_freq_cumulative.png', bbox_inches = "tight")
plt.close()




########## Loughran McDonald Sentiment Word List

#This will allow us to do the sentiment analysis. Let's first load the word list below.
sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']
#
df_sentiment = pd.read_csv('loughran_mcdonald_master_dic_2016.csv')
df_sentiment.columns = [column.lower() for column in df_sentiment.columns] # Lowercase the columns for ease of use
#
# Remove unused information
df_sentiment = df_sentiment[sentiments + ['word']]
df_sentiment[sentiments] = df_sentiment[sentiments].astype(bool)
df_sentiment = df_sentiment[(df_sentiment[sentiments]).any(1)]
#
# Apply the preprocessing to these words
df_sentiment['word'] = lemmatize_words(df_sentiment['word'].str.lower())
df_sentiment         = df_sentiment.drop_duplicates('word')
#
print(df_sentiment.head())
'''
    negative  positive  uncertainty  litigious  constraining  interesting          word
9       True     False        False      False         False        False       abandon
12      True     False        False      False         False        False   abandonment
13      True     False        False      False         False        False  abandonments
51      True     False        False      False         False        False      abdicate
54      True     False        False      False         False        False    abdication
'''
df_sentiment.to_csv('df_sentiment.csv', header=True, index=False)




########## Sentiment Analysis

##### Merge two data frame by 'word'

#print(pd.merge(df_sorted, df_sentiment, on='word', how='inner'))

df_sorted_sentiment = pd.merge(df_sorted, df_sentiment, on='word', how='inner')
df_sorted_sentiment.to_csv('df_sorted_sentiment.csv', header=True, index=False)