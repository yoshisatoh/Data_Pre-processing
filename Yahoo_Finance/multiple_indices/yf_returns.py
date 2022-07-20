#################### Data Pre-processing: Stock Return Calculation by using Close prices on Yahoo Finance  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/04/04
# Last Updated: 2022/07/20
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/multiple_indices/yf_returns.py
#
#
########## Input Data Files
#
'''
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
########## Usage Instructions
#
#Run this py script on Terminal of MacOS (or Command Prompt of Windows) as follows:
'''
##### 2. Calculate daily index returns
python yf_returns.py "^GSPC"
python yf_returns.py "^IXIC"
python yf_returns.py "^RUT"
python yf_returns.py "ZT=F"
python yf_returns.py "ZF=F"
python yf_returns.py "ZN=F"
python yf_returns.py "ZB=F"
python yf_returns.py "CL=F"
python yf_returns.py "NG=F"
python yf_returns.py "GC=F"
python yf_returns.py "HG=F"
python yf_returns.py "ZC=F"
python yf_returns.py "ZS=F"
'''
#
#
########## Output Data File(s)
'''
##### 2. Calculate daily index returns
y_Returns_^GSPC.csv
y_Returns_^IXIC.csv
y_Returns_^RUT.csv
y_Returns_ZT=F.csv
y_Returns_ZF=F.csv
y_Returns_ZN=F.csv
y_Returns_ZB=F.csv
y_Returns_CL=F.csv
y_Returns_NG=F.csv
y_Returns_GC=F.csv
y_Returns_HG=F.csv
y_Returns_ZC=F.csv
y_Returns_ZS=F.csv
'''
#
#
########## References
#
#
#
####################################################################################################




########## install Python libraries
#
# pip on your Terminal on MacOS (or Command Prompt on Windows) might not work.
#pip install pandas
#
# If that's the case, then try:
#pip install --upgrade pandas --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
# If it's successful, then you can repeat the same command for other libraries (e.g., numpy).
#
#
########## import Python libraries
#
import sys
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt




########## arguments
for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

arg_ticker     = str(sys.argv[1])       #'^GSPC'
arg_ticker_csv = arg_ticker + '.csv'    #'^GSPC.csv'




########## Calculate daily returns

df_EQ = pd.read_csv(arg_ticker_csv, usecols =['Date', 'Close'])
#print(df_EQ)

df_EQ_Return = df_EQ['Close'].pct_change()
#print(df_EQ_Return)

df_EQ_Return = pd.merge(df_EQ['Date'], df_EQ_Return, how='outer', left_index=True, right_index=True)
#print(df_EQ_Return)

df_EQ_Return.rename(columns={'Close': arg_ticker}, inplace=True)
print(df_EQ_Return)

df_EQ_Return.to_csv('y_Returns_' + arg_ticker_csv, sep=',', header=True, index=False)



