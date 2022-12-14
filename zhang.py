import os
import pandas as pd  
import numpy as np
np.set_printoptions(suppress=True) #不輸出科學記號

# 輸出文件夾下的所有文件
n=[]
n=os.listdir(r'D:\data\OptionsDaily')
n2=[]
n2=os.listdir(r'D:\data\FuturesDaily')

ans=np.zeros((len(n),4))

for i, j in zip(range(134,135), range(134,135)):
    test = pd.read_csv('D:/data/OptionsDaily/'+n[i],usecols=[1,2,4,6,7])
    x= test['          商品代號']=='    TXO     '
    txo=test[x]
    y=txo['        買賣權別']=='    C     '
    y1=txo['        買賣權別']=='    P     '
    txo_c=txo[y]
    txo_p=txo[y1]
    K_=pd.DataFrame([txo_c['        履約價格']]).T
    C_=pd.DataFrame([txo_c['          成交價格']]).T
    K_p=pd.DataFrame([txo_p['        履約價格']]).T
    P_=pd.DataFrame([txo_p['          成交價格']]).T
    count1=pd.DataFrame([txo_c['         成交數量(B or S)']]).T
    count2=pd.DataFrame([txo_p['         成交數量(B or S)']]).T
    sum_1=int(len(txo_c['        履約價格']))
    sum_2=int(len(txo_p['        履約價格']))
    test2 = pd.read_csv('D:/data/FuturesDaily/'+n2[j],encoding='big5',usecols=[1,4])
    x= test2['商品代號'] == 'TX     '
    txo=test2[x]
    F_=pd.DataFrame([txo['成交價格']]).T
    print(i)
    print(j)
    ans[i]=counting()
    print(i)
    print(j)
    
	
An= pd.DataFrame(ans)   
An.columns=['iv_call','skewness_call','iv_put','skewness_put']

def counting():
        T=10/365
        rf=0.0119
        MktPrice = np.array(C_[0:sum_1],dtype=np.float)
        K = np.array(K_[0:sum_1],dtype=np.float)
        T = np.array(T,dtype=np.float)
        volume = np.array(count1[0:sum_1],dtype=np.float)
        S = np.array(F_[0:sum_1],dtype=np.float)
        x0 = [5,0.02,0.01,0.01,-0.5]
        bnds = [[1e-8,30],[1e-8,2],[1e-8,2],[1e-8,2],[-1,1]]	
        x,w = GenerateGaussLegendre(32)
        from scipy.optimize import minimize
        sol = minimize(Heston_Loss_Function_slsqp,x0,
                       args=(S,rf,MktPrice,K,T,volume,x,w),method='slsqp',
                       bounds=bnds)
        #while sol['fun'] >= 100 or sol['success']==False:
        x0 = [np.random.uniform(2,7),np.random.uniform(0,0.1),
                      np.random.uniform(0,0.1),np.random.uniform(0,0.1),
                      np.random.uniform(-0.8,-0.1)]
        sol = minimize(Heston_Loss_Function_slsqp,x0,
                               args=(S,rf,MktPrice,K,T,volume,x,w),
                               method='slsqp',bounds=bnds, 
                               options={'ftol': 1e-01})          
        param = sol['x']
        x0 = param
        params=[]
        params.append(param)
        #error = sol['fun']
        #print(' 誤差: ',sol['fun'],' 是否成功:',sol['success'])
        Heston_skew_list=Heston_skew(param[0],param[1],param[2],0,param[3],
                                     param[4],7/365)
        call_v=param[3]
        call_sk=Heston_skew_list
        
        MktPrice = np.array(P_[0:sum_2],dtype=np.float)
        K = np.array(K_p[0:sum_2],dtype=np.float)
        T = np.array(T,dtype=np.float)
        volume = np.array(count2[0:sum_2],dtype=np.float)
        S = np.array(F_[0:sum_2],dtype=np.float)
        x0 = [5,0.02,0.01,0.01,-0.5]
        bnds = [[1e-8,30],[1e-8,2],[1e-8,2],[1e-8,2],[-1,1]]	
        x,w = GenerateGaussLegendre(32)
        from scipy.optimize import minimize
        sol = minimize(Heston_Loss_Function_slsqp,x0,
                       args=(S,rf,MktPrice,K,T,volume,x,w),method='slsqp',
                       bounds=bnds)
        #while sol['success']==False:
        x0 = [np.random.uniform(2,7),np.random.uniform(0,0.1),
                      np.random.uniform(0,0.1),np.random.uniform(0,0.1),
                      np.random.uniform(-0.8,-0.1)]
        sol = minimize(Heston_Loss_Function_slsqp,x0,
                               args=(S,rf,MktPrice,K,T,volume,x,w),
                               method='slsqp',bounds=bnds, 
                               options={'ftol': 1e-01})          
        param = sol['x']
        x0 = param
        params=[]
        params.append(param)
        #error = sol['fun']
        #print(' 誤差: ',sol['fun'],' 是否成功:',sol['success'])
        Heston_skew_list=Heston_skew(param[0],param[1],param[2],0,param[3],
                                     param[4],7/365)
        put_v=param[3]
        put_sk=Heston_skew_list
        return call_v,call_sk,put_v,put_sk
        

def GenerateGaussLegendre(n):
    m = np.int(np.floor(n/2))
    L = np.zeros(m+1)
    for k in range(m+1):
        L[k] = pow(0.5,n)*pow(-1,k)*np.math.factorial(2*n-2*k)/np.math.factorial(k)/np.math.factorial(n-k)/np.math.factorial(n-2*k)

    P = np.zeros(n+1)
    for k in range(n+1):
        if np.mod(k+1,2)==0:
            P[k]=0
        else:
            P[k] = L[np.int(k/2)]

    x = np.sort(np.roots(P))
    w = np.zeros(n)
    dC = np.zeros([m+1,n])
    for j in range(n):
        for k in range(m+1):
            dC[k,j] = pow(0.5,n)*pow(-1,k)*np.math.factorial(2*n-2*k)/np.math.factorial(k)/np.math.factorial(n-k)/np.math.factorial(n-2*k)*(n-2*k)*pow(x[j],(n-2*k-1))
    
        w[j] = 2/(1-pow(x[j],2))/pow(np.sum(dC[:,j]),2)
    return x,w

def Heston_Loss_Function_slsqp(param,S,rf,MktPrice,K,T,volume,x,w):
    kappa  = param[0]
    theta  = param[1]
    sigma  = param[2]
    v0     = param[3]
    rho    = param[4]
    lambd = 0
    
    #volume_sum = np.sum(volume)
    ModelPrice = np.array([])
    #w_vol = np.array([])
    for i in range(len(MktPrice)):
        ModelPrice = np.append(ModelPrice,Heston(S[i],K[i],T,rf,kappa,theta,sigma,lambd,v0,rho,x,w))
        #w_vol = np.append(w_vol,volume[i]/volume_sum)

    #error = ((pow(MktPrice - ModelPrice,2)/MktPrice)).sum()
    error = ((pow(MktPrice - ModelPrice,2))).sum()
    return error/len(MktPrice)


def Heston(S,K,T,rf,kappa,theta,sigma,lambd,v0,rho,x,w,a=0,b=1000):
    int1 = np.zeros(len(x))
    int2 = np.zeros(len(x))
    
    for k in range(len(x)):
        X = (a+b)/2 + (b-a)/2*x[k]
        int1[k] = w[k]*HestonProb(X,kappa,theta,lambd,rho,sigma,T,S,K,rf,v0,1)
        int2[k] = w[k]*HestonProb(X,kappa,theta,lambd,rho,sigma,T,S,K,rf,v0,2)
    
    P1 = 0.5 + 1/np.pi*np.sum(int1)*(b-a)/2
    P2 = 0.5 + 1/np.pi*np.sum(int2)*(b-a)/2
    
    C = S*P1 - K*np.exp(-rf*T)*P2
    return C

def HestonProb(phi,kappa,theta,lambd,rho,sigma,tau,S,K,rf,v0,Pnum):
    x = np.log(S)
    a = kappa * theta
    if Pnum == 1:
        w = 0.5
        b = kappa + lambd - rho*sigma
    elif Pnum ==2:
        w = -0.5
        b = kappa + lambd
    
    d = np.sqrt((1j*phi*rho*sigma-b)**2 - pow(sigma,2)*(2*1j*phi*w-phi*phi))
    
    g = (b - 1j*phi*rho*sigma + d)/(b - 1j*phi*rho*sigma - d)
    
    C = rf*phi*1j*tau + ((b-rho*sigma*phi*1j+d)*tau - 2*np.log((1-g*np.exp(d*tau))/(1-g)))*a/pow(sigma,2)
    D = ((1-np.exp(d*tau))/(1-g*np.exp(d*tau))) * (b-rho*sigma*phi*1j+d)/pow(sigma,2)
    
    final = np.exp(-1j*phi*np.log(K))*np.exp(C + D*v0 + 1j*phi*x)/1j/phi
    return final.real

def Heston_skew(kappa,theta,sigma,lambd,v0,rho,tau):
    e1kt = np.exp(kappa*tau)
    e2kt = np.exp(2*kappa*tau)
    e3kt = np.exp(3*kappa*tau)
    sigma2 = pow(sigma,2)
    sigma3 = pow(sigma,3)
    kappa2 = pow(kappa,2)
    kappa3 = pow(kappa,3)
    kappa4 = pow(kappa,4)
    A = 6*e3kt*v0*sigma3 - 22*e3kt*sigma3*theta + 3*e2kt*v0*sigma3\
        + 15*e2kt*sigma3*theta + 24*e1kt*kappa2*v0*rho*sigma2*tau\
        - 12*e1kt*kappa2*rho*sigma2*tau*theta - 12*e1kt*kappa*v0*sigma3*tau\
        + 6*e1kt*kappa*sigma3*tau*theta + 36*e1kt*kappa*v0*rho*sigma2\
        - 24*e1kt*kappa*rho*sigma2*theta - 6*e1kt*v0*sigma3 + 6*e1kt*sigma3*theta\
        - 3*v0*sigma3 + sigma3*theta - 24*e1kt*kappa2*v0*sigma + 12*e1kt*kappa2*sigma*theta\
        - 48*e3kt*kappa3*v0*rho + 96*e3kt*kappa3*rho*theta + 24*e3kt*kappa2*v0*sigma\
        - 60*e3kt*kappa2*sigma*theta + 48*e2kt*kappa3*v0*rho - 96*e2kt*kappa3*rho*theta\
        + 48*e2kt*kappa2*sigma*theta - 6*e2kt*kappa2*v0*sigma3*tau*tau\
        + 6*e2kt*kappa2*sigma3*tau*tau*theta + 48*e3kt*kappa2*v0*rho*rho*sigma\
        - 144*e3kt*kappa2*rho*rho*sigma*theta + 6*e3kt*kappa*sigma3*tau*theta\
        - 36*e3kt*kappa*v0*rho*sigma2 + 120*e3kt*kappa*rho*sigma2*theta\
        - 48*e2kt*kappa2*v0*rho*rho*sigma + 144*e2kt*kappa2*rho*rho*sigma*theta\
        - 6*e2kt*kappa*v0*sigma3*tau + 18*e2kt*kappa*sigma3*tau*theta - 48*e3kt*kappa4*rho*tau*theta\
        - 96*e2kt*kappa*rho*sigma2*theta + 24*e3kt*kappa3*sigma*tau*theta + 48*e2kt*kappa4*v0*rho*tau\
        - 48*e2kt*kappa4*rho*tau*theta - 48*e2kt*kappa3*v0*sigma*tau \
        + 48*e2kt*kappa3*sigma*tau*theta - 24*e2kt*kappa4*v0*rho*rho*sigma*tau*tau\
        + 24*e2kt*kappa4*rho*rho*sigma*tau*tau*theta + 48*e3kt*kappa3*rho*rho*sigma*tau*theta\
        + 24*e2kt*kappa3*v0*rho*sigma2*tau*tau - 24*e2kt*kappa3*rho*sigma2*tau*tau*theta\
        - 36*e3kt*kappa2*rho*sigma2*tau*theta - 48*e2kt*kappa3*v0*rho*rho*sigma*tau\
        + 96*e2kt*kappa3*rho*rho*sigma*tau*theta + 48*e2kt*kappa2*v0*rho*sigma2*tau\
        - 96*e2kt*kappa2*rho*sigma2*tau*theta
    
    B = -8*e2kt*kappa2*rho*sigma*tau*theta + 8*e2kt*kappa3*tau*theta + 2*e2kt*kappa*sigma2*tau*theta\
        + 8*e1kt*kappa2*v0*rho*sigma*tau - 8*e1kt*kappa2*rho*sigma*tau*theta\
        - 8*e2kt*kappa*v0*rho*sigma + 16*e2kt*kappa*rho*sigma*theta - 4*e1kt*kappa*v0*sigma2*tau\
        + 4*e1kt*kappa*sigma2*tau*theta + 8*e2kt*kappa2*v0 - 8*e2kt*kappa2*theta\
        + 2*e2kt*v0*sigma2 - 5*e2kt*sigma2*theta + 8*e1kt*kappa*v0*rho*sigma\
        - 16*e1kt*kappa*rho*sigma*theta - 8*e1kt*kappa2*v0 + 8*e1kt*kappa2*theta\
        + 4*e1kt*sigma2*theta - 2*v0*sigma2 + sigma2*theta
        
    skew = -sigma*np.sqrt(2)*A/np.sqrt(kappa)/pow(B,1.5)
    return skew



import pandas_datareader.data as web
import fix_yahoo_finance
'''
技術指標
'''
#大盤資料
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

import talib
#RSI
RSI_=talib.RSI(np.array(close), timeperiod=12) 
rsi=pd.DataFrame(RSI_)
rsi.index=idx
rsi=rsi.sort_index(ascending=False)
rsi_=rsi[0:len(n)]
rsi_.columns=['RSI']

#KD
kd = talib.STOCH(high=high_['High'], 
                low=low_['Low'], 
                close=close_['Close'],
                )
kd_=pd.DataFrame([kd[0],kd[1]]).T
kd_=kd_.sort_index(ascending=False)
kd_=kd_[0:len(n)]
kd_['KDminus'] = kd[0] - kd[1]
kd_.columns=['K','D','KDminus']

#MACD
'''
macd = 12天EMA - 26天EMA 
signal = 9天MACD的EMA 
hist = MACD - MACD signal
'''
macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
macd_=pd.DataFrame([macd,macdsignal,macdhist]).T
macd_.columns=['macd', 'macd_signal', 'macd_hist']
macd_=macd_.sort_index(ascending=False)
macd_=macd_[0:len(n)]

#MA
ma5=talib.SMA(close,timeperiod=5)
ma20=talib.SMA(close,timeperiod=20)
ma_=pd.DataFrame([ma5,ma20]).T
ma_.columns=['ma5', 'ma20']
ma_=ma_.sort_index(ascending=False)
ma_=ma_[0:len(n)]

'''
把結果讀進DataFrame，加上index(日期)
'''   
zhang=zhang.sort_index(ascending=False)
x=rsi_.index
zhang.index=x

'''
計算大盤return
'''
fix_yahoo_finance.pdr_override()
start = '2018-11-01'
end = '2019-06-01'
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

data=data.sort_index(ascending=False)

'''
把所有技術指標寫進同一個DataFrame
'''
ta_result=pd.concat([rsi_,kd_,macd_,ma_],axis=1)

'''
GC_result,ta_result讀進同一個DataFrame
'''
data_result2=pd.concat([zhang,ta_result,data],axis=1)
data_result2=data_result2.fillna(0)
data_result2 = data_result2.drop(data_result2.index[:1])

'''
returns計算勝率(0 or 1)
'''
w=pd.DataFrame(data_result2['returns'])
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
x=data_result2.index
return_.index=x

'''
讀進同一個DataFrame
'''
data_result2=pd.concat([data_result2,return_],axis=1)
#型態轉為float
data_result2=pd.DataFrame(data_result2,dtype=np.float)
data_result2.to_csv(r"C:\Users\Emily\Desktop\20190617_Emily\data_result2.csv")


'''
迴歸分析(call)
'''
w=pd.DataFrame(data_result2['iv_call'])
w1=pd.DataFrame(data_result2['skewness_call'])

a=[]
a1=[]
w_=pd.DataFrame(data_result2['iv_put'])
w1_=pd.DataFrame(data_result2['skewness_put'])

a__=[]
a1_=[]
aa=[]
re = data_result2['returns']
r_=[]
r=[]
a_iv=[]
for i in range(0,len(w)-1):
    a.append(float((w.iloc[1+i])-(w.iloc[i])))
    a1.append(float((w1.iloc[1+i])-(w1.iloc[i])))
    a__.append(float((w_.iloc[1+i])-(w_.iloc[i])))
    a1_.append(float((w1_.iloc[1+i])-(w1_.iloc[i])))
    #aa.append(float((w.iloc[1+i])-(w.iloc[i]))*(w1.iloc[1+i])-(w1.iloc[i])))
for i in range(0,len(a)):
    aa.append(float(a__[i]*a1_[i]))
    a_iv.append(float(a[i]*a1[i]))
    r.append(float(re.iloc[i]))

a_=pd.DataFrame([a,a1,a_iv,a__,a1_,aa,r]).T
a_.columns=['iv_call_', 'skewness_call_','ivs_call','iv_put_', 'skewness_put_','ivs_put','R']

for i in range(0,len(a)):
    r_.append(float((re.iloc[1+i])-(re.iloc[i])))
    
rr_=pd.DataFrame([r_]).T
rr_.columns=['returns']

da=pd.concat([a_,rr_],axis=1)

import statsmodels.formula.api as sm
model1 = sm.ols(formula = 'R ~ iv_call_ + skewness_call_ ',data = da).fit()
print(model1.summary()) 
model2 = sm.ols(formula = 'R ~ iv_call_ + skewness_call_ +ivs_call',data = da).fit()
print(model2.summary()) 
model3 = sm.ols(formula = 'returns ~ iv_call_ + skewness_call_ ',data = da).fit()
print(model3.summary()) 
model4 = sm.ols(formula = 'returns ~ iv_call_ + skewness_call_ +ivs_call',data = da).fit()
print(model4.summary()) 


'''
迴歸分析(put)
'''
model1 = sm.ols(formula = 'R ~ iv_put_ + skewness_put_ ',data = da).fit()
print(model1.summary()) 
model2 = sm.ols(formula = 'R ~ iv_put_ + skewness_put_ +ivs_put',data = da).fit()
print(model2.summary()) 
model3 = sm.ols(formula = 'returns ~ iv_put_ + skewness_put_ ',data = da).fit()
print(model3.summary()) 
model4 = sm.ols(formula = 'returns ~ iv_put_ + skewness_put_ +ivs_put',data = da).fit()
print(model4.summary()) 


'''
迴歸預測
'''
import seaborn as sns
#X是想探索的自變數，Y是依變數。
#準備X & y array
X = data_result2[['iv_call','skewness_call','RSI','KDminus','macd_hist','returns']]
y = data_result2['returns']

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


'''
DNN預測
'''
from keras.layers import Dense, LSTM, Conv1D
from keras.models import Sequential
import tensorflow as tf
from keras.utils import np_utils  
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt  

def DNN_model(input_size, cell_sizes, dropout, activation_fn):
    model = Sequential()
    model.add(Dense(cell_sizes[0],input_shape=input_size,activation=activation_fn))
    [model.add(Dense(units = i, activation=activation_fn)) for i in cell_sizes[1:]]
    model.add(Dense(3,
                    kernel_regularizer=l2(0.01), 
                    bias_regularizer=None, 
                    activity_regularizer=l1(0.01),
                    activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def show_train_history(train_history, train, validation):  
    plt.figure()
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 

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