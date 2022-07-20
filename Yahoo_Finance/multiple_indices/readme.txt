


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


##### 3. Aggregate daily index returns into one file
python yf_returns_agg.py "y_Returns_^GSPC.csv" "y_Returns_^IXIC.csv" "y_Returns_^RUT.csv" "y_Returns_ZT=F.csv" "y_Returns_ZF=F.csv" "y_Returns_ZN=F.csv" "y_Returns_ZB=F.csv" "y_Returns_CL=F.csv" "y_Returns_NG=F.csv" "y_Returns_GC=F.csv" "y_Returns_HG=F.csv" "y_Returns_ZC=F.csv" "y_Returns_ZS=F.csv"


##### See the final daily return-series file:
##### y.csv