#################### Data Pre-processing: Aggregating Stock Returns  ####################
#
#  (C) 2022, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2022/04/04
# Last Updated: 2022/07/20
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/Yahoo_Finance/multiple_indices/yf_returns_agg.py
#
#
########## Input Data Files
#
'''
y_Returns_^GSPC.csv    #SP 500
y_Returns_^IXIC.csv    #NASDAQ Composite
y_Returns_^RUT.csv     #Russell 2000
y_Returns_ZT=F.csv     #2-Year T-Note Futures
y_Returns_ZF=F.csv     #Five-Year US Treasury Note Futu
y_Returns_ZN=F.csv     #10-Year T-Note Futures
y_Returns_ZB=F.csv     #U.S. Treasury Bond Futures
y_Returns_CL=F.csv     #Crude Oil
y_Returns_NG=F.csv     #Natural Gas
y_Returns_GC=F.csv     #Gold
y_Returns_HG=F.csv     #Copper
y_Returns_ZC=F.csv     #Corn Futures
y_Returns_ZS=F.csv     #Soybean Futures
'''
#
#
########## Usage Instructions
#
#Run this py script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python yf_returns_agg.py "y_Returns_^GSPC.csv" "y_Returns_^IXIC.csv" "y_Returns_^RUT.csv" "y_Returns_ZT=F.csv" "y_Returns_ZF=F.csv" "y_Returns_ZN=F.csv" "y_Returns_ZB=F.csv" "y_Returns_CL=F.csv" "y_Returns_NG=F.csv" "y_Returns_GC=F.csv" "y_Returns_HG=F.csv" "y_Returns_ZC=F.csv" "y_Returns_ZS=F.csv"
#
#
########## Output Data File(s)
#
#y.csv
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
#
#print(len(sys.argv))
#print(len(sys.argv)-1)
#exit()


#arg_ticker_csv_1     = str(sys.argv[1])       #"y_Returns_^GSPC.csv"
#arg_ticker_csv_2     = str(sys.argv[2])       #"y_Returns_^DJI.csv"
#arg_ticker_csv_3     = str(sys.argv[3])       #"y_Returns_^IXIC.csv"
#
for i in range(len(sys.argv)):
    exec(f"arg_ticker_csv_{i} = str(sys.argv[i])")
    exec(f"print(arg_ticker_csv_{i})")
print("")
#
#exit()




########## Calculate daily returns

#print(arg_ticker_csv_1)

#df_EQ_Return_1 = pd.read_csv(arg_ticker_csv_1, index_col='Date')
#df_EQ_Return_2 = pd.read_csv(arg_ticker_csv_2, index_col='Date')
#df_EQ_Return_3 = pd.read_csv(arg_ticker_csv_3, index_col='Date')
#print(df_EQ_Return_1)
#
#for i in range(len(sys.argv)):
for i in range(1, len(sys.argv)):
    exec(f"df_EQ_Return_{i} = pd.read_csv(arg_ticker_csv_{i}, index_col='Date')")
    #
    ###exec(f"csvfname = arg_ticker_csv_{i}")
    ###print(csvfname)
    ###exec(f"df_EQ_Return_{i} = pd.read_csv(csvfname, index_col='Date')")
    #
    exec(f"print(df_EQ_Return_{i})")
    #
print("")
#exit()



#df_EQ_Returns  = pd.merge(df_EQ_Return_1, df_EQ_Return_2, how='outer', left_index=True, right_index=True)
#df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_3, how='outer', left_index=True, right_index=True)
#print(df_EQ_Returns.dropna())

for i in range(2, len(sys.argv)):
    if i == 2:
        #df_EQ_Returns  = pd.merge(df_EQ_Return_1, df_EQ_Return_2, how='outer', left_index=True, right_index=True)
        exec(f"df_EQ_Returns  = pd.merge(df_EQ_Return_{i-1},  df_EQ_Return_{i}, how='outer', left_index=True, right_index=True)")
    else:
        #df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_3, how='outer', left_index=True, right_index=True)
        exec(f"df_EQ_Returns  = pd.merge(df_EQ_Returns,  df_EQ_Return_{i}, how='outer', left_index=True, right_index=True)")

df_EQ_Returns  = df_EQ_Returns.dropna()
print(df_EQ_Returns)

df_EQ_Returns.to_csv('y.csv', sep=',', header=True, index=True)



