#################### Data Pre-processing: Stock Prices on Yahoo Finance ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/11/03
# Last Updated: 2021/11/03
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sentiment/hybrid/multi_class/finviz_Twitter/2.1.0.yf.py
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
#python 2.1.0.yf.py TSLA 2021-09-30 2021-11-01 30m
#Note:
#Yahoo Finance usually uses Time zone in New York, NY, USA (GMT-4 when Eastern Daylight/Standard Time), but two dates as arguments are based on YOUR LOCAL TIME ZONE.
#If you would like to get monthly data in October 2021, then you might want to have a margin of time as above.
#
#Generally,
#python 2.1.0.yf.py (arg_ticker: a ticker on Yahoo Finance) (arg_start) (arg_end) (arg_interval)
#
#arg_ticker: a stock ticker, e.g., TSLA
#
#arg_start, arg_end: yyyy-mm-dd
#
#arg_interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')
#
#
########## Output Data File(s)
#
#2.1.1.ticker_history.csv
#2.1.2.ticker_history.csv
#2.1.3.ticker_history.csv
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
#python -m pip install yfinance ==0.1.64
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade yfinance  --trusted-host pypi.org --trusted-host files.pythonhosted.org



########## import Python libraries

import sys

import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

import datetime




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #2.1.0.yf.py

arg_ticker   = str(sys.argv[1])    #'TSLA'

arg_start    = str(sys.argv[2])    #"2021-10-01",
arg_end      = str(sys.argv[3])    #"2021-10-29",
arg_interval = str(sys.argv[4])    #'30m'    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')




########## Getting stock prices

ticker = yf.Ticker(arg_ticker)

ticker_history = ticker.history(
    start    = arg_start,
    end      = arg_end,
    interval = arg_interval
).reset_index()

ticker_history.to_csv('2.1.1.ticker_history.csv', header=True, index=False)

print(ticker_history.head())
'''
                   Datetime        Open        High         Low       Close   Volume  Dividends  Stock Splits
0 2021-09-30 10:30:00-04:00  784.599976  787.289978  778.700012  787.229980        0          0             0
1 2021-09-30 11:30:00-04:00  787.299988  787.459900  778.330078  779.340027  2198545          0             0
2 2021-09-30 12:30:00-04:00  779.270020  785.640015  777.770020  783.200012  1622335          0             0
3 2021-09-30 13:30:00-04:00  783.049988  789.000000  782.109985  786.500000  1843967          0             0
4 2021-09-30 14:30:00-04:00  786.494995  787.389893  778.500000  779.426025  1979808          0             0
'''




########## Getting intraday and closing prices on a daily basis

#print(ticker_history[['Datetime', 'Open', 'Close', 'Volume']].head())
#
#pandas DataFrame
#print(ticker_history[['Datetime']].head())
#print(type(ticker_history[['Datetime']]))
#
#pandas Series
#print(ticker_history['Datetime'])
#print(type(ticker_history['Datetime']))
#
#print(ticker_history[['Datetime']])
#print(str(ticker_history[['Datetime']]).split())
#
#print(type(ticker_history['Datetime']))

########## Data pre-processing
#
ticker_history['date'] = pd.to_datetime(ticker_history['Datetime']).dt.date

ticker_history['time'] = pd.to_datetime(ticker_history['Datetime']).dt.time
#
ticker_history['price'] = ticker_history['Open']
#
#print(ticker_history['time'][0])    #11:00:00
#print(type(ticker_history['time'][0]))    #<class 'datetime.time'>
#print(ticker_history['time'][0].hour)    #11
#print(ticker_history['time'][0].minute)    #0
#print(ticker_history['time'][0].second)    #0
#
#
#If 'time' is 15:30:00, then 'Close' is the closing price of the day
print(len(ticker_history))
for i in range(len(ticker_history)):
     if (ticker_history['time'][i].hour == 15) and (ticker_history['time'][i].minute == 30) and (ticker_history['time'][i].second == 0):
         print(str(ticker_history['time'][i]) + ' ' + str(ticker_history['Close'][i]))
         #print(ticker_history.append({'Datetime': ticker_history['Datetime'][i], 'Open': ticker_history['Open'][i], 'High': ticker_history['High'][i], 'Low': ticker_history['Low'][i], 'Close': ticker_history['Close'][i], 'Volume': ticker_history['Volume'][i], 'Dividends': ticker_history['Dividends'][i], 'Stock Splits': ticker_history['Stock Splits'][i], 'date': ticker_history['date'][i], 'time': '16:00:00', 'price': ticker_history['Close'][i]}, ignore_index=True))
         ticker_history = ticker_history.append({'Datetime': ticker_history['Datetime'][i], 'Open': ticker_history['Open'][i], 'High': ticker_history['High'][i], 'Low': ticker_history['Low'][i], 'Close': ticker_history['Close'][i], 'Volume': ticker_history['Volume'][i], 'Dividends': ticker_history['Dividends'][i], 'Stock Splits': ticker_history['Stock Splits'][i], 'date': ticker_history['date'][i], 'time': '16:00:00', 'price': ticker_history['Close'][i]}, ignore_index=True)
#
ticker_history = ticker_history.sort_values(['date', 'time'])
#
#
#print(type(ticker_history['time']))    #<class 'pandas.core.series.Series'>
#
#print(pd.to_datetime(ticker_history['Datetime']).dt.time == pd.to_datetime('15:30:00', format='%H:%M:%S'))
#ticker_history['close_frag'] = pd.to_datetime(ticker_history['Datetime']).dt.time == pd.to_datetime('15:30:00', format='%H:%M:%S')
#ticker_history['close_frag'] = pd.to_datetime(ticker_history['Datetime']).dt.time == pd.to_datetime('15:30:00', format='%H:%M:%S')
#
#print(ticker_history.index)
#print(type(ticker_history.index))
#print(ticker_history['Datetime'].at_time(datetime.time(15, 30, 0)))    #15:30:00
#
print(ticker_history.head())
#
ticker_history.to_csv('2.1.2.ticker_history.csv', header=True, index=False)




########## Final stock price data pre-processing

ticker_history = pd.read_csv('2.1.2.ticker_history.csv', usecols=['date', 'time', 'price'], header=0, index_col=False)

#print(pd.to_datetime(ticker_history['date'], format="%Y-%m-%d"))
ticker_history['date'] = pd.to_datetime(ticker_history['date'], format="%Y-%m-%d")

print(ticker_history.head())

ticker_history.to_csv('2.1.3.ticker_history.csv', header=True, index=False, date_format='%Y-%m-%d')
