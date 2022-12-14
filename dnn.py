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