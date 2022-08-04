#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis (Embedding-based, Binary Class, Data on Twitter) ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/08/04
# Last Updated: 2022/08/04
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/single_stock_financials/yff.py
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
#python yff.py MSFT
#
#Generally,
#python yf.py (arg_ticker: a ticker on Yahoo Finance)
#
#arg_ticker: a stock ticker e.g., MSFT
#
#
#
#
########## Output Data File(s)
#
#balance_sheet.csv
#calendar.csv
#cashflow.csv
#earningst.csv
#financials.csv
#info.csv
#institutional_holders.csv
#major_holders.csv
#news.csv
#options.csv
#quarterly_balance_sheet.csv
#quarterly_cashflow.csv
#quarterly_earningst.csv
#quarterly_financials.csv
#recommendations.csv
#sustainability.csv
#
#
#
#
########## References
#
#yfinance 0.1.74
#https://pypi.org/project/yfinance/
#
#
####################




########## install Python libraries (before running this script)
#
#pip install yfinance  --upgrade
#pip install yfinance  --U
#python -m pip install yfinance 
#python -m pip install yfinance ==0.1.74
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade yfinance  --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## import Python libraries

import sys

import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

import csv




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #yff.py

arg_ticker   = str(sys.argv[1])    #"MSFT"




########## Getting stock prices

ticker = yf.Ticker(arg_ticker)

# stock info
#print(ticker.info)
#
#print(type(ticker.info))
#<class 'dict'>
#
'''
#items in a horizontal direction
#using DictWriter
with open('info.csv', 'w', newline='', encoding='utf-8') as f:  # This is for Python 3.x. You need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, ticker.info.keys())
    w.writeheader()
    w.writerow(ticker.info)
'''
'''
#items in a horizontal direction
with open('info.csv','w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(ticker.info.keys())
    w.writerow(ticker.info.values())
'''
#
#items in a vertical direction
with open('info.csv','w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerows(ticker.info.items())
#




# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
#####print(ticker.isin)


# Dividends,Stock Splits
#####print(ticker.actions)
#####print(ticker.dividends)
#####print(ticker.splits)


# show financials
#print(ticker.financials)    # annual, 4 years
#print(type(ticker.financials))
#<class 'pandas.core.frame.DataFrame'>
ticker.financials.to_csv('financials.csv', header=True, index=True)
#
#
#print(ticker.quarterly_financials)    # quarterly, 1 year
#print(type(ticker.quarterly_financials))
#<class 'pandas.core.frame.DataFrame'>
ticker.quarterly_financials.to_csv('quarterly_financials.csv', header=True, index=True)
#
#
# show balance sheet
#print(ticker.balance_sheet)    # annual, 4 years
ticker.balance_sheet.to_csv('balance_sheet.csv', header=True, index=True)
#
#print(ticker.quarterly_balance_sheet)    # quarterly, 1 year
ticker.quarterly_balance_sheet.to_csv('quarterly_balance_sheet.csv', header=True, index=True)
#
#
# show earnings
#print(ticker.earnings)    # annual, 4 years
ticker.earnings.transpose().to_csv('earningst.csv', header=True, index=True)
#
#print(ticker.quarterly_earnings)    # quarterly, 1 year
ticker.quarterly_earnings.transpose().to_csv('quarterly_earningst.csv', header=True, index=True)
#
#
# show cashflow
#print(ticker.cashflow)    # annual, 4 years
ticker.cashflow.to_csv('cashflow.csv', header=True, index=True)
#
#print(ticker.quarterly_cashflow)    # quarterly, 1 year
ticker.quarterly_cashflow.to_csv('quarterly_cashflow.csv', header=True, index=True)
#
#
# show all earnings dates
#####print(ticker.earnings_dates)
#####AttributeError: 'Ticker' object has no attribute 'earnings_dates'




# show major holders
#print(ticker.major_holders)
#print(type(ticker.major_holders))
#<class 'pandas.core.frame.DataFrame'>
ticker.major_holders.to_csv('major_holders.csv', header=False, index=False)


# show institutional holders
#print(ticker.institutional_holders)
ticker.institutional_holders.to_csv('institutional_holders.csv', header=True, index=False)


# show sustainability
#print(ticker.sustainability)
ticker.sustainability.to_csv('sustainability.csv', header=True, index=True)


# show analysts recommendations
print(ticker.recommendations)
ticker.recommendations.to_csv('recommendations.csv', header=True, index=True)


# show next event (earnings, etc)
print(ticker.calendar)
ticker.calendar.to_csv('calendar.csv', header=False, index=True)


# show options expirations
#print(ticker.options)
#print(type(ticker.options))
#<class 'tuple'>
'''
with open('options.csv','w') as out:
    csv_out = csv.writer(out)
    #csv_out.writerow(['name','num'])
    #for row in ticker.options:
    #    csv_out.writerow(row)
    # You can also do csv_out.writerows(data) instead of the for loop
    csv_out.writerows(ticker.options)
'''


# get option chain for specific expiration
#
#####opt = ticker.option_chain('YYYY-MM-DD')
#####ValueError: Expiration `YYYY-MM-DD` cannot be found. Available expiration are: [2022-08-05, 2022-08-12, 2022-08-19, 2022-08-26, 2022-09-02, 2022-09-09, 2022-09-16, 2022-10-21, 2022-11-18, 2022-12-16, 2023-01-20, 2023-02-17, 2023-03-17, 2023-06-16, 2023-09-15, 2024-01-19, 2024-06-21]
#
#opt = ticker.option_chain('2024-06-21')
#print(opt)
# data available via: opt.calls, opt.puts

# show news
#print(ticker.news)
#print(type(ticker.news))
#<class 'list'>
'''
with open('news.csv', 'w', newline='', encoding='utf-8') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    #write.writerow(header)
    write.writerows(ticker.news)
'''
df = pd.DataFrame(ticker.news) 
df.to_csv('news.csv') 

