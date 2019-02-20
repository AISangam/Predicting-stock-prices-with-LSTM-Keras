import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
dataset = pandas.read_csv('TESLA STOCK.csv')
dataset = dataset.drop(dataset.index[0])
dataset = dataset.drop(['date'], axis=1)
data = dataset.iloc[:,1:]
target = dataset.iloc[:,0]
data=data.values
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
target =target.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
print("Please show me the shape of X_train:",X_train.shape)
print("Please show me the shape of X_test:",X_test.shape)
print("Please show me the shape of y_train:",y_train.shape)
print("Please show me the shape of y_:",y_test.shape)
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
model = Sequential()
model.add(LSTM(64, input_shape=(1,4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1)
Predict = model.predict(X_test)
testScore = math.sqrt(mean_squared_error(y_test, Predict))
print('Test Score: %.2f RMSE' % (testScore))
plt.plot(y_test)
plt.plot(Predict)
plt.legend(['original value','predicted value'],loc='upper right')
plt.show()