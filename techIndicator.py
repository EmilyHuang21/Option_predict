import pandas as pd  
import numpy as np
import pandas_datareader.data as web
import fix_yahoo_finance
import scipy.stats as ss

'''
#計算歷史波動偏態峰態(初始值)
'''
def history_skew_kurtosis(startDate, endDate) :
	fix_yahoo_finance.pdr_override()
	start = startDate
	end = endDate
	tickers = ['^TWII']
	Asset = web.get_data_yahoo(tickers, start, end)
	close = Asset['Adj Close']
	# calculate daily returns
	daily_returns = np.log(close/close.shift(1))
	daily_returns2 = daily_returns.dropna()
	data=pd.DataFrame(daily_returns)
	data.columns=['returns']
	data2=pd.DataFrame(daily_returns2)
	data2.columns=['returns']
	sigma=np.std(data2['returns'])
	skew=ss.skew(data2['returns'])
	kurt=ss.kurtosis(data2['returns'])
	
	return data,sigma,skew,kurt

'''
技術指標 Technical Indicators
'''
#大盤資料 stock market index
def stockIndex(startDate, endDate)
	fix_yahoo_finance.pdr_override()
	start = '2018-09-01'
	end = '2019-05-31'
	tickers = ['^TWII']
	Asset = web.get_data_yahoo(tickers, start, end)
	close = Asset['Close']
	close_=pd.DataFrame(close)
	adjclose = Asset['Adj Close']
	adjclose_=pd.DataFrame(adjclose)
	high = Asset['High']
	high_=pd.DataFrame(high)
	low = Asset['Low']
	low_=pd.DataFrame(low)
	idx = Asset.index
	
	return close,close_,adjclose,adjclose_,high,high_,low,low_,idx

import talib
#RSI
def _RSI_(close,idx):
	RSI_=talib.RSI(np.array(close), timeperiod=12) 
	rsi=pd.DataFrame(RSI_)
	rsi.index=idx
	rsi=rsi.sort_index(ascending=False)
	rsi_=rsi[0:len(y_f)]
	rsi_.columns=['RSI']
	return rsi_

#KD
def _KD_(high_,low_,close_):
	kd = talib.STOCH(high=high_['High'], 
					low=low_['Low'], 
					close=close_['Close'],
					)
	kd_=pd.DataFrame([kd[0],kd[1]]).T
	kd_=kd_.sort_index(ascending=False)
	kd_=kd_[0:len(y_f)]
	kd_['KDminus'] = kd[0] - kd[1]
	kd_.columns=['K','D','KDminus']
	return kd_

#MACD
'''
macd = 12天EMA - 26天EMA 
signal = 9天MACD的EMA 
hist = MACD - MACD signal
'''
def _macd_(close):
	macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
	macd_=pd.DataFrame([macd,macdsignal,macdhist]).T
	macd_.columns=['macd', 'macd_signal', 'macd_hist']
	macd_=macd_.sort_index(ascending=False)
	macd_=macd_[0:len(y_f)]
	return macd_

#MA
def _ma_(close):
	ma5=talib.SMA(close,timeperiod=5)
	ma20=talib.SMA(close,timeperiod=20)
	ma_=pd.DataFrame([ma5,ma20]).T
	ma_.columns=['ma5', 'ma20']
	ma_=ma_.sort_index(ascending=False)
	ma_=ma_[0:len(y_f)]
	return ma_