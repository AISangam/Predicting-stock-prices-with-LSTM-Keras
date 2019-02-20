#pandas is imported to read the csv file and perform preprocessing on the dataset.
import pandas
#matlpotlib is used to visualize the plot
import matplotlib.pyplot as plt
#MinMaxScalar is used to normalize the value before training
from sklearn.preprocessing import MinMaxScaler
#numoy is used to deal with the data after train and split as data will be in form of aray for training and testing.
import numpy as np
#keras has 2 models one is functional and another is sequential
from keras.models import Sequential
#Dense layer is the output layer
from keras.layers import Dense
#LSTM is Long Term Short Term Memory
from keras.layers import LSTM
#library used to calculate the mean square error. For classification accuracy is calculated and for regression mean square error is calculated
import math
from sklearn.metrics import mean_squared_error
#data is read using pandas and output is a dataframe
dataset = pandas.read_csv('TESLA STOCK.csv')
#index is dropped
dataset = dataset.drop(dataset.index[0])
#date axis is dropped using drop function
dataset = dataset.drop(['date'], axis=1)
#iloc is used for index where loc is used for label
data = dataset.iloc[:,1:]
target = dataset.iloc[:,0]
#convert dataframe in numpy array
data=data.values
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
target =target.values
#module for training and splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
print("Please show me the shape of X_train:",X_train.shape)
print("Please show me the shape of X_test:",X_test.shape)
print("Please show me the shape of y_train:",y_train.shape)
print("Please show me the shape of y_:",y_test.shape)
#converting the shape in the way machine will take for training
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
model = Sequential()
#46 neurons will be used at this layer
model.add(LSTM(64, input_shape=(1,4)))
#output layer contains 1 neuron to predict the output
model.add(Dense(1))
#as the data is continous, hence loss function is mean_squared_error
model.compile(loss='mean_squared_error', optimizer='sgd')
#data is trained here
model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1)
#prediction using X_test
Predict = model.predict(X_test)
testScore = math.sqrt(mean_squared_error(y_test, Predict))
print('Test Score: %.2f RMSE' % (testScore))
plt.plot(y_test)
plt.plot(Predict)
plt.legend(['original value','predicted value'],loc='upper right')
plt.show()
