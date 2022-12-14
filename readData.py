import os
#import csv 
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
import pandas_datareader.data as web
import fix_yahoo_finance
import scipy.stats as ss
import scipy.optimize as opt
def read_intraday():
	# 輸出文件夾下的所有文件
	n=[]
	n=os.listdir(r'D:\Emily\data\OptionsDaily')
	K_=[]
	C_=[]
	K_p=[]
	P_=[]
	count=[]
	for i in range(0, len(n)):
		test = pd.read_csv('D:/Emily/data/OptionsDaily/'+n[i],usecols=[1,2,4,6,7])
		x= test['          商品代號'] == '    TXO     '
		txo=test[x]
		y=txo['        買賣權別']=='    C     '
		y1=txo['        買賣權別']=='    P     '
		txo_c=txo[y]
		txo_p=txo[y1]
		K_.append(float(np.mean(txo_c['        履約價格'])))
		C_.append(float(np.mean(txo_c['          成交價格'])))
		count.append(float(len(txo_c['        履約價格']+txo_p['        履約價格'])))
		K_p.append(float(np.mean(txo_p['        履約價格'])))
		P_.append(float(np.mean(txo_p['          成交價格'])))
		print(i)
	for i in range(0, len(n2)):
    test = pd.read_csv('D:/Emily/data/FuturesDaily/'+n2[i],encoding='big5',usecols=[1,4])
    x= test['商品代號'] == 'TX     '
    txo=test[x]
    F_.append(float(np.mean(txo['成交價格'])))
    print(i)

	T=1/360
	r=0.019

	#把資料讀進dataframe
	y_k=pd.DataFrame([K_]).T
	y_c=pd.DataFrame([C_]).T
	y_f=pd.DataFrame([F_]).T
	y_kp=pd.DataFrame([K_p]).T
	y_p=pd.DataFrame([P_]).T
	
	#計算選擇權與期貨日內資料的基本統計量
	d1 = pd.Series(C_)
	d2 = pd.Series(K_)
	d3 = pd.Series(F_)
	d4 = pd.Series(P_)
	d5 = pd.Series(K_p)
	
	df = pd.DataFrame(np.array([d1,d2,d3,d4,d5]).T, columns=['Call','履約價_call','期貨','Put','履約價_put'])
	df.head()
	df=df.apply(status)
	df_count=pd.DataFrame((count), columns=['筆數'])
	eq=np.mean(df_count)
	df_count.plot()
	plt.show()
	return df,d1,d2,d3,d4,d5



def status(x) : 
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['總天數','最小值','最小值位置','25%分位數', '中位數','75%分位數','均值','最大值','最大值位數','平均絕對偏差','方差','標準差','偏度','峰度'])

#顯示(繪圖)排出不合理價格的選擇權日內交易資料每天的總筆數
import matplotlib.pyplot as plt

