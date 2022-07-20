#################### Data Pre-processing: Stock Prices on Yahoo Finance ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/28
# Last Updated: 2022/07/20
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/multiple_indices/yf.py
#
########## Input Data File(s)
#
#N/A
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
'''
##### 1. Get index prices
python yf.py "^GSPC" 2012-03-31 2022-07-19 1d    #SP 500
python yf.py "^IXIC" 2012-03-31 2022-07-19 1d    #NASDAQ Composite
python yf.py "^RUT"  2012-03-31 2022-07-19 1d    #Russell 2000
python yf.py "ZT=F"  2012-03-31 2022-07-19 1d    #ZT=F	2-Year T-Note Futures
python yf.py "ZF=F"  2012-03-31 2022-07-19 1d    #ZF=F	Five-Year US Treasury Note Futu
python yf.py "ZN=F"  2012-03-31 2022-07-19 1d    #ZN=F	10-Year T-Note Futures
python yf.py "ZB=F"  2012-03-31 2022-07-19 1d    #ZB=F	U.S. Treasury Bond Futures
python yf.py "CL=F"  2012-03-31 2022-07-19 1d    #CL=F	Crude Oil
python yf.py "NG=F"  2012-03-31 2022-07-19 1d    #NG=F	Natural Gas
python yf.py "GC=F"  2012-03-31 2022-07-19 1d    #GC=F	Gold
python yf.py "HG=F"  2012-03-31 2022-07-19 1d    #HG=F	Copper
python yf.py "ZC=F"  2012-03-31 2022-07-19 1d    #ZC=F	Corn Futures
python yf.py "ZS=F"  2012-03-31 2022-07-19 1d    #ZS=F	Soybean Futures
'''
#
#Generally,
#python yf.py (arg_ticker: a ticker on Yahoo Finance) (arg_start) (arg_end) (arg_interval)
#
#arg_ticker: a ticker e.g., "^GSPC"
#
#arg_start, arg_end: yyyy-mm-dd
#
#arg_interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')
#
#
########## Output Data File(s)
#
'''
##### 1. Get index prices
^GSPC.csv    #SP 500
^IXIC.csv    #NASDAQ Composite
^RUT.csv     #Russell 2000
ZT=F.csv     #2-Year T-Note Futures
ZF=F.csv     #Five-Year US Treasury Note Futu
ZN=F.csv     #10-Year T-Note Futures
ZB=F.csv     #U.S. Treasury Bond Futures
CL=F.csv     #Crude Oil
NG=F.csv     #Natural Gas
GC=F.csv     #Gold
HG=F.csv     #Copper
ZC=F.csv     #Corn Futures
ZS=F.csv     #Soybean Futures
'''
#
#
########## References
#
#yfinance 0.1.64
#https://pypi.org/project/yfinance/
#
#Yahoo! Finance - World Indices
#https://finance.yahoo.com/world-indices
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
#import matplotlib.pyplot as plt

import yfinance as yf




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #yf.py

arg_ticker   = str(sys.argv[1])    #'^GSPC'
arg_start    = str(sys.argv[2])    #"2012-03-31",
arg_end      = str(sys.argv[3])    #"2022-07-19",
arg_interval = str(sys.argv[4])    #'1d'    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (optional, default is '1d')




########## Getting stock prices

ticker = yf.Ticker(arg_ticker)

ticker_history = ticker.history(
    start    = arg_start,
    end      = arg_end,
    interval = arg_interval
).reset_index()

ticker_history.to_csv(arg_ticker + '.csv', header=True, index=False)
