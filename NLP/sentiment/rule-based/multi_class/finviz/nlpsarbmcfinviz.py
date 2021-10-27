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
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sentiment/rule-based/multi_class/finviz/nlpsarbmcfinviz.py
#
#
########## Input Data File(s)
#
#tickers.txt
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlpsarbmcfinviz.py tickers.txt
#
#Generally,
#python nlpsarbmcfinviz.py (a txt file for a list of tickers)
#
#
########## Output Data File(s)
#
#parsed_news.txt
#parsed_and_scored_news.csv
#mean_scores.csv
#mean_scores_raw.csv
#
#
########## References
#
#Sentiment Analysis of Stocks from Financial News using Python
#https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
#
#
#
####################




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




########## import Python libraries

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finwiz_url = 'https://finviz.com/quote.ashx?t='

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

#print(sys.argv[0])    #nlpsarbmc.py

arg_filename_tickers = str(sys.argv[1])    #tickers.txt




########## 0. Rule-based sentiment analysis

'''
The main drawback with the rule-based approach for sentiment analysis is that the method only cares about individual words and completely ignores the context in which it is used. 

For example, “the party was savage” will be negative when considered by any token-based algorithms.
'''




##### 0.1. VADER sentiment

'''
Valence aware dictionary for sentiment reasoning (VADER) is another popular rule-based sentiment analyzer. 

It uses a list of lexical features (e.g. word) which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment.   

Vader sentiment returns the probability of a given input sentence to be 

positive, negative, and neutral. 

Vader is optimized for social media data and can yield good results when used with data from twitter, facebook, etc.
'''

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

str_sentence = 'The food was great!'

vs = analyzer.polarity_scores(str_sentence)
print("{:-<65} {}".format(str_sentence, str(vs)))    #{'compound': 0.6588, 'neg': 0.0, 'neu': 0.406, 'pos': 0.594}


##### 0.2. Textblob
'''
It is a simple python library that offers API access to different NLP tasks such as sentiment analysis, spelling correction, etc.

Textblob sentiment analyzer returns two properties for a given input sentence: 

Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments. 
Subjectivity is also a float which lies in the range of [0,1]. Subjective sentences with higher number like 1 generally refer to personal opinion, emotion, or judgment. 

Textblob will ignore the words that it doesn’t know, it will consider words and phrases that it can assign polarity to and averages to get the final score.
'''
from textblob import TextBlob

str_sentence = 'The food was great!'

testimonial = TextBlob(str_sentence)
print(testimonial.sentiment)    #Sentiment(polarity=1.0, subjectivity=0.75)




########## 1. Import Libraries

# See the "import Python libraries" section above.




########## 2. Store the Date, Time and News Headlines Data


news_tables = {}


#tickers = ['AMZN', 'TSLA', 'GOOG']
#
with open(arg_filename_tickers, "r") as tf:
    tickers = tf.read().split('\n')
#    
for ticker in tickers:
    print(ticker)
#
print(tickers)
print(type(tickers))




for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    # Add the table to our dictionary
    news_tables[ticker] = news_table




########## 3. Print the Data Stored in news_tables (optional)

# <tr></tr> tags - rows (individual records)
# <a></a> tags - headlines
# <td></td> tags - date and time

# Read one single day of headlines for the first stock e.g., 'AMZN' 

stck0 = news_tables[tickers[0]]


# Get all the table rows tagged in HTML with <tr> into 'stck0_tr'
stck0_tr = stck0.findAll('tr')

for i, table_row in enumerate(stck0_tr):
    # Read the text of the element 'a' into 'link_text'
    a_text = table_row.a.text
    # Read the text of the element 'td' into 'data_text'
    td_text = table_row.td.text
    # Print the contents of 'link_text' and 'data_text' 
    print(a_text)
    print(td_text)
    # Exit after printing 4 rows of data
    if i == 3:
        break




########## 4. Parse the Date, Time and News Headlines into a Python List

'''
The following code is similar to the one above, but this time it parses the date, time and headlines into a Python list called parsed_news instead of printing it out. The if, else loop is necessary because if you look at the news headlines above, only the first news of each day has the ‘date’ label, the rest of the news only has the ‘time’ label so we have to account for this.
'''

parsed_news = []

# Iterate through the news
for file_name, news_table in news_tables.items():
    # Iterate through all tr tags in 'news_table'
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Extract the ticker from the file name, get the string up to the 1st '_'  
        ticker = file_name.split('_')[0]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, text])
        
#parsed_news
print(parsed_news)

f = open('parsed_news.txt', 'w')
for x in parsed_news:
    f.write(str(x) + "\n")
f.close()




########## 5. Sentiment Analysis with Vader!

# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Set column names
columns = ['ticker', 'date', 'time', 'headline']

# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

# Convert the date column from string to datetime
parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

parsed_and_scored_news.head()

parsed_and_scored_news.to_csv('parsed_and_scored_news.csv', header=True, index=False)
'''
The first 5 rows of the DataFrame from the code above should look something like this. The ‘compound’ column gives the sentiment scores. For positive scores, the higher the value, the more positive the sentiment is. Similarly for negative scores, the more negative the value, the more negative the sentiment is. The scores range from -1 to 1.
'''




########## 6. Plot a Bar Chart of the Sentiment Score for Each Day

plt.rcParams['figure.figsize'] = [12, 6]

# Group by date and ticker columns from scored_news and calculate the mean
mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()


print(mean_scores.head())
mean_scores_raw = mean_scores
mean_scores_raw.to_csv('mean_scores_raw.csv', header=True, index=True)


# Unstack the column ticker
mean_scores = mean_scores.unstack()


# Get the cross-section of compound in the 'columns' axis
mean_scores = mean_scores.xs('compound', axis="columns").transpose()

# Plot a bar chart with pandas
mean_scores.plot(kind = 'bar')

#print(type(mean_scores))
mean_scores.to_csv('mean_scores.csv', header=True, index=True)

plt.grid()
plt.ylabel('compound')
plt.savefig('Fig.png')
plt.show()