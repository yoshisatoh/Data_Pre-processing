#################### Data Pre-processing: Calculating Cumulative Returns  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/07/22
# Last Updated: 2022/07/22
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/multiple_indices/yf_cum_returns_agg.py
#
#
########## Input Data Files
#
'''
y.csv    # Daily Returns of Multiple Investment Instruments
'''
#
#
########## Usage Instructions
#
#Run this py script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python yf_cum_returns_agg.py y.csv
#
#
########## Output Data File(s)
#
#ycumprod.csv
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
print("")

#print(str(sys.argv[0]))       #"yf_cum_returns_agg.py"
#print(str(sys.argv[1]))       #"y.csv"


########## Calculate daily cumulative returns

df = pd.read_csv(str(sys.argv[1]), index_col='Date')

df_cumprod = (1 + df).cumprod()
df_cumprod = df_cumprod.dropna()
df_cumprod.to_csv('ycumprod.csv', sep=',', header=True, index=True)

