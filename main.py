import os
#import csv 
import pandas as pd  
import numpy as np
import pandas_datareader.data as web
import fix_yahoo_finance
import scipy.stats as ss
import scipy.optimize as opt

df_intraday = read_intraday()
startDate = '2018-11-01'
endDate = '2019-06-01'
history_data[]= history_skew_kurtosis(startDate, endDate)

x1=np.zeros((len(y_f),3))
for i in range(0,len(y_k)):
    K=float(y_k.iloc[i])
    C=float(y_c.iloc[i])
    F=float(y_f.iloc[i])
    print(i)
    im=[sigma,skew,kurt]
    bnds=((0.0001,0.5),(-3.0,3.0),(-3.0,10.0))
    im=opt.minimize(implied_call, x0=im, method='SLSQP'
                    ,bounds=bnds,options={'ftol': 1e-07})
    x1[i]=im['x']

x2=np.zeros((len(y_f),3))
for i in range(0,len(y_k)):
    K=float(y_kp.iloc[i])
    P=float(y_p.iloc[i])
    F=float(y_f.iloc[i])
    print(i)
    im=[sigma,skew,kurt]
    bnds=((0.001,0.5),(-3.0,3.0),(-3.0,10.0))
    im=opt.minimize(implied_put, x0=im, method='SLSQP'
                    ,bounds=bnds,options={'ftol': 1e-07})
    x2[i]=im['x']

'''
技術指標
'''
#大盤資料
start = '2018-09-01'
end = '2019-05-31'
df_stock[]=stockIndex(startDate, endDate)

rsi_ = _rsi_(df_stock['close'],df_stock['idx'])
kd_ = _kd_(df_stock['high'],df_stock['low'],df_stock['close'])
macd_ = _macd_(df_stock['close'])
ma_ = _rsi(df_stock['close'])

'''
把結果讀進DataFrame，加上index(日期)
'''   
GC_result1=pd.DataFrame(x1)
GC_result1.columns=['iv_call','skewness_call','kurtosis_call']
GC_result2=pd.DataFrame(x2)
GC_result2.columns=['iv_put','skewness_put','kurtosis_put']
GC_result=pd.concat([GC_result1,GC_result2],axis=1)
GC_result=GC_result.sort_index(ascending=False)
x=rsi_.index
GC_result.index=x

data=data.sort_index(ascending=False)

'''
把所有技術指標寫進同一個DataFrame
'''
ta_result=pd.concat([rsi_,kd_,macd_,ma_],axis=1)

'''
GC_result,ta_result讀進同一個DataFrame
'''
data_result=pd.concat([GC_result,ta_result,data],axis=1)
data_result=data_result.fillna(0)
data_result = data_result.drop(data_result.index[:1])

'''
returns計算勝率(0 or 1)
'''
w=pd.DataFrame(data_result['returns'])
a=[]
for i in range(0,len(w)-1):
    q=float((w.iloc[1+i])-(w.iloc[i]))
    if q > 0.0005:
        a.append('1')
    else:
        a.append('0')
a_=pd.DataFrame(a)
first=pd.DataFrame([0])
return_=pd.concat([first,a_],axis=0)
return_.columns=['returns_']
x=data_result.index
return_.index=x

'''
讀進同一個DataFrame
'''
data_result=pd.concat([data_result,return_],axis=1)
#型態轉為float
data_result=pd.DataFrame(data_result,dtype=np.float)
#data_result.info()
#dtypes: float64(17)
data_result.to_csv(r"D:\Emily\data_result.csv")

'''
迴歸分析
'''
w=pd.DataFrame(data_result['iv_call'])
w1=pd.DataFrame(data_result['skewness_call'])
w2=pd.DataFrame(data_result['kurtosis_call'])
a=[]
a1=[]
a2=[]
w_=pd.DataFrame(data_result['iv_put'])
w1_=pd.DataFrame(data_result['skewness_put'])
w2_=pd.DataFrame(data_result['kurtosis_put'])
a__=[]
a1_=[]
a2_=[]
aa=[]
re = data_result['returns']
r_=[]
r=[]
a_iv=[]
for i in range(0,len(w)-1):
    a.append(float((w.iloc[1+i])-(w.iloc[i])))
    a1.append(float((w1.iloc[1+i])-(w1.iloc[i])))
    a2.append(float((w2.iloc[1+i])-(w2.iloc[i])))
    a__.append(float((w_.iloc[1+i])-(w_.iloc[i])))
    a1_.append(float((w1_.iloc[1+i])-(w1_.iloc[i])))
    a2_.append(float((w2_.iloc[1+i])-(w2_.iloc[i])))
    #aa.append(float((w.iloc[1+i])-(w.iloc[i]))*(w1.iloc[1+i])-(w1.iloc[i])))
for i in range(0,len(a)):
    aa.append(float(a__[i]*a1_[i]))
    a_iv.append(float(a[i]*a1[i]))
    r.append(float(re.iloc[i]))
    
a_=pd.DataFrame([a,a1,a2,a_iv,a__,a1_,a2_,aa,r]).T
a_.columns=['iv_call_', 'skewness_call_','kurtosis_call_','ivs_call','iv_put_', 'skewness_put_','kurtosis_put_','ivs_put','R']

for i in range(0,len(a)):
    r_.append(float((re.iloc[1+i])-(re.iloc[i])))
    
rr_=pd.DataFrame([r_]).T
rr_.columns=['returns']

da=pd.concat([a_,rr_],axis=1)

#買權迴歸
import statsmodels.formula.api as sm
model1 = sm.ols(formula = 'R ~ iv_call_ + skewness_call_ +kurtosis_call_',data = da).fit()
print(model1.summary()) 
model2 = sm.ols(formula = 'R ~ iv_call_ + skewness_call_ +kurtosis_call_+ivs_call',data = da).fit()
print(model2.summary()) 
model3 = sm.ols(formula = 'returns ~ iv_call_ + skewness_call_ +kurtosis_call_',data = da).fit()
print(model3.summary()) 
model4 = sm.ols(formula = 'returns ~ iv_call_ + skewness_call_ +kurtosis_call_+ivs_call',data = da).fit()
print(model4.summary()) 

#賣權迴歸
model1 = sm.ols(formula = 'R ~ iv_put_ + skewness_put_ +kurtosis_put_',data = da).fit()
print(model1.summary()) 
model2 = sm.ols(formula = 'R ~ iv_put_ + skewness_put_ +kurtosis_put_+ivs_put',data = da).fit()
print(model2.summary()) 
model3 = sm.ols(formula = 'returns ~ iv_put_ + skewness_put_ +kurtosis_put_',data = da).fit()
print(model3.summary()) 
model4 = sm.ols(formula = 'returns ~ iv_put_ + skewness_put_ +kurtosis_put_+ivs_put',data = da).fit()
print(model4.summary()) 


'''
迴歸預測
'''
import seaborn as sns
#X是想探索的自變數，Y是依變數。
#準備X & y array
X = data_result[['iv_call','skewness_call','kurtosis_call','RSI','KDminus','macd_hist','returns']]
y = data_result['returns']

#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split
#test_size代表測試組比例。random_state代表設定隨機種子，讓測試結果可被重複
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#載入線性迴歸，並訓練模型
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#使用測試組資料來預測結果
predictions = lm.predict(X_test)
predictions


from keras.utils import np_utils  
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt  

np.random.seed(12345)

X=data_result[['returns']]
y=data_result['returns_']

train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X.iloc[0:train_size,:], X.iloc[train_size:len(X),:]
y_train, y_test = y.iloc[0:train_size], y.iloc[train_size:len(X)]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#pre = preprocessing.MinMaxScaler().fit(X_train)
y_TrainOneHot = np_utils.to_categorical(y_train+1)
y_TestOneHot = np_utils.to_categorical(y_test+1)
look_back = 3
epochs = 100
batch_size=5
cells = 4
layers = 3
dropout = 1
activation_fn = 'sigmoid' #'relu','sigmoid','tanh'

cell_sizes = [cells for i in range(layers)]

model = DNN_model(input_size=(np.shape(X)[1],),cell_sizes=cell_sizes,
                  dropout=dropout,activation_fn=activation_fn)
train_history = model.fit(X_train.as_matrix(),y_TrainOneHot,batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
Train_cost, Train_acc = model.evaluate(X_train.as_matrix(), y_TrainOneHot, batch_size=batch_size)

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
Test_cost, Test_acc = model.evaluate(X_test.as_matrix(), y_TestOneHot, batch_size=batch_size)

#預測Predict
trainPredict = model.predict(y_train)
testPredict = model.predict(y_test)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(y_train)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data_result)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(y)-1, :] = testPredict
