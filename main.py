import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.metrics import mean_squared_error

import math


def create_dataset(dataset, look_back=1, look_forward=0):
    """Returns two arrays of the variable described in dataset: one at t and
    the other at t+1. """
    dataX0, dataX1 = [], []
    for i in range(len(dataset)-look_back-look_forward-1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[i + look_back + look_forward, 0]
        dataX0.append(a)
        dataX1.append(b)
    return np.array(dataX0), np.array(dataX1)


# # Upload the data and show them

df = pd.read_csv('elevation.csv', sep=';', header=None,
                 names=['Index', 'Time', 'Elevation (in meters)'])

# # Format the dataframe

# Drop the first column
df = df.drop(df.columns[[0]], axis=1)

# Reformat the dates
df['Time'] = df['Time'].map(lambda x: str(x)[:-11])
df['Time'] = pd.to_datetime(df['Time'])

# Change the index as the date
time = df.iloc[:, 0].values
df = df.set_index('Time')
print(df.head())

# Conversion of integers as floating
df = df.astype('float32')

# Show graph
df.plot(grid=True)
plt.show()

# Fix random seed to ensure reproductability
np.random.seed(2308)

# # Normalization of the dataset and nparray conversion

scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df.astype(float))

# Splitting of training and testing sets
training_size = int(len(df) * 0.70)
testing_size = len(df) - training_size
training, testing = df[0:training_size, :], df[training_size:, :]


# # Create two datasets of the elevation at t and t+1 for the training and the
# # testing sets to be able to construct the memory blocks.


look_back = 15  # predictions from the 5 previous days
look_forward = 10  # predictions on the 10 next days
trX0, trX1 = create_dataset(training, look_back, look_forward)
teX0, teX1 = create_dataset(testing, look_back, look_forward)

# # Reshape the input for model.fit use

trainingX0 = np.reshape(trX0, (trX0.shape[0], 1, trX0.shape[1]))
testingX0 = np.reshape(teX0, (teX0.shape[0], 1, teX0.shape[1]))

# # Model

# Create and fit the model

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))  # 2 LSTM layers
model.add(Dense(1))  # output layer (linear)
model.compile(loss='mean_squared_error', optimizer='adam')
# mean_squared_error as objective function and Adam optimizer

history = model.fit(trainingX0, trX1, validation_data=(testingX0, teX1),
                    epochs=20, batch_size=10, verbose=2)

# Make predictions

trainPredict = model.predict(trainingX0)
testPredict = model.predict(testingX0)

# Invert predictions (meter units)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trX1])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([teX1])

# Calculate RMSE

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# Evaluate the model
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

# Plot training history
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.legend()
plt.show()

# Plots

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+(look_forward*2)+1:len(df)-1, :] = testPredict

# plot baseline and predictions
plt.plot(time, scaler.inverse_transform(df), 'k')
plt.plot(time, trainPredictPlot)
plt.plot(time, testPredictPlot)
plt.show()
