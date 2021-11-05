#################### Data Pre-processing: News, finviz ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/11/03
# Last Updated: 2021/11/03
#
# Github:
# https://github.com/yoshisatoh/CFA/blob/main/2.2.0.finviz.py
#
#
########## Input Data File(s)
#
#2.2.0.tickers.txt
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python 2.2.0.finviz.py 2.2.0.tickers.txt
#
#Generally,
#python 2.2.0.finviz.py (a txt file for a list of tickers)
#
#
########## Output Data File(s)
#
#2.2.1.parsed_news.csv
#2.2.2.df_parsed_news.csv
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

import datetime
#from datetime import date
#from datetime import datetime

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

arg_filename_tickers = str(sys.argv[1])    #2.2.0.tickers.txt




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
        #print(type(text))    #<class 'str'>
        text = text.replace(',','')    #remove commans in text
        #
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
#print(parsed_news)
print(type(parsed_news))    #<class 'list'>

f = open('2.2.1.parsed_news.csv', 'w')
for x in parsed_news:
    f.write(str(x).replace('[', '').replace(']', '') + "\n")
f.close()




########## 5. Clean, Sort

#df_parsed_news = pd.read_csv('2.2.1.parsed_news.csv', sep=',', quotechar="'", names=['ticker', 'date', 'time', 'text'])
df_parsed_news = pd.read_csv('2.2.1.parsed_news.csv', sep=',', names=['ticker', 'date', 'time', 'text'])


#Remove single quotes in the ticker, date, and time columns
df_parsed_news['ticker'] = df_parsed_news['ticker'].str.replace("'", "")
df_parsed_news['date'] = df_parsed_news['date'].str.replace("'", "")
df_parsed_news['time'] = df_parsed_news['time'].str.replace("'", "")
df_parsed_news['time'] = df_parsed_news['time'].str.replace(" ", "")
df_parsed_news['text'] = df_parsed_news['text'].str.replace(" '", "'")


#print(pd.to_datetime(df_parsed_news['date']))    #format='%Y-%m-%d'
df_parsed_news['date'] = pd.to_datetime(df_parsed_news['date'])


#print(datetime(df_parsed_news['time']))    #format='%H:%M:%S'
#print(pd.to_datetime(df_parsed_news['time'], format='%H:%M:%S'))
print(datetime.date.strftime(pd.to_datetime("04:00PM", format='%I:%M%p'), format="%H:%M:%S"))
#
for i in range(len(df_parsed_news['time'])):
    #print(datetime.date.strftime(pd.to_datetime(df_parsed_news['time'][0], format='%I:%M%p'), format="%H:%M:%S"))
    #print(datetime.date.strftime(pd.to_datetime(df_parsed_news['time'][i], format='%I:%M%p'), format="%H:%M:%S"))
    df_parsed_news['time'][i] = datetime.date.strftime(pd.to_datetime(df_parsed_news['time'][i], format='%I:%M%p'), format="%H:%M:%S")


#sort by ticker, date, and time in ascending order
df_parsed_news = df_parsed_news.sort_values(['ticker', 'date', 'time'])

print(df_parsed_news.head())

df_parsed_news.to_csv('2.2.2.df_parsed_news.csv', header=True, index=False, date_format='%Y-%m-%d')