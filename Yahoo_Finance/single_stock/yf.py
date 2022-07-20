#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis (Embedding-based, Binary Class, Data on Twitter) ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/27
# Last Updated: 2022/07/20
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/single_stock/yf.py
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
#python yf.py TSLA 2020-10-28 2021-10-27 1d
#
#Generally,
#python yf.py (arg_ticker: a ticker on Yahoo Finance) (arg_start) (arg_end) (arg_interval)
#
#arg_ticker: a stock ticker e.g., TSLA
#
#arg_start, arg_end: yyyy-mm-dd
#
#arg_interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')
#
#
########## Output Data File(s)
#
#ticker_history.csv
#
#
########## References
#
#yfinance 0.1.64
#https://pypi.org/project/yfinance/
#
#
####################




########## install Python libraries (before running this script)
#
#pip install yfinance  --upgrade
#pip install yfinance  --U
#python -m pip install yfinance 
#python -m pip install yfinance ==0.17.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade yfinance  --trusted-host pypi.org --trusted-host files.pythonhosted.org



########## import Python libraries

import sys

import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #yf.py

arg_ticker   = str(sys.argv[1])    #'TSLA'

arg_start    = str(sys.argv[2])    #"2020-10-28",
arg_end      = str(sys.argv[3])    #"2021-10-27",
arg_interval = str(sys.argv[4])    #'1d'    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')




########## Getting stock prices

ticker = yf.Ticker(arg_ticker)

ticker_history = ticker.history(
    start    = arg_start,
    end      = arg_end,
    interval = arg_interval
).reset_index()

ticker_history.to_csv('ticker_history.csv', header=True, index=False)
#
#ticker_history.csv:
'''
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2020-10-27,423.760009765625,430.5,420.1000061035156,424.67999267578125,22686500,0,0
...
2021-10-26,1024.68994140625,1094.93994140625,1001.4400024414062,1018.4299926757812,62415000,0,0
'''