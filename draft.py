import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.metrics import mean_squared_error

import math

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

# Convert integer as floating, which is more convenient for neural networks
# modelling.
df = df.astype('float32')

# Show graph
# df.plot(grid=True)
# plt.show()

# Fix random seed to ensure reproductability
np.random.seed(7)
# However, if you have an algorithm that is based in random numbers (e.g. a NN), reproducibility may be a problem when you want to share your results. Someone that re-runs your code will be ensured to get different results, as randomness is part of the algorithm. But, you can tell the random number generator to instead of starting from a seed taken randomly, to start from a fixed seed. That will ensure that while the numbers generated are random between themseves, they are the same each time (e.g. [3 84 12 21 43 6] could be the random output, but ti will always be the same).

# # Normalization of the dataset - more practical when using sigmoid fonction for the outputs and for the sake of comparaison for the inputs
# Not dataframe because not as good to normalize

scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df.astype(float))  # type becomes nparray

# Split the training (70%) and testing (30%) sets >> common
training_size = int(len(df) * 0.70)
testing_size = len(df) - training_size
training, testing = df[0:training_size, :], df[training_size:, :]


# # Create two datasets of the elevation at t and t+1 for the training and the
# # testing sets to be able to construct the memory blocks.

def create_dataset(dataset, look_back=1):
    """Returns two arrays of the variable described in dataset: one at t and
    the other at t+1. """
    dataX0, dataX1 = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX0.append(a)
        dataX1.append(dataset[i + look_back, 0])
    return np.array(dataX0), np.array(dataX1)


look_back = 5
  # predict the value at the next time in the sequence (t+1), we can use the current time (t), as well as the 14 prior times as input variables.
trX0, trX1 = create_dataset(training, look_back)
teX0, teX1 = create_dataset(testing, look_back)

print(trX0, trX1)

# # Reshape the input >>> WORK ON INTUITION: goal - have [[[x]]] >> list of Numpy arrays asked for the model.fit

trainingX0 = np.reshape(trX0, (trX0.shape[0], 1, trX0.shape[1]))
testingX0 = np.reshape(teX0, (teX0.shape[0], 1, teX0.shape[1]))

# # Model

# Create and fit the model
# By default learning rate of 0.01 >> try to define if another one not better

model = Sequential()  # linear stack of layers model
model.add(LSTM(2, input_shape=(1, look_back)))  # 1 input and 2 hidden layers, tanh activation function by default
# keep tan h function as (1) Often found to converge faster in practice (2) Gradient computation is less expensive
model.add(Dense(1))  # last layer
# output shape of 1 (elevation) - implements the operation: output = activation(dot(input, kernel) + bias) - linear activation function by default
model.compile(loss='mean_squared_error', optimizer='adam')  # configure the learning process
# loss: objective that the model will try to minimize - function defined as MSE
# adam because the best and many features:
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# https://arxiv.org/abs/1412.6980v8

history = model.fit(trainingX0, trX1, validation_data=(testingX0, teX1),
                    epochs=100, batch_size=10, verbose=2)
# trainingX0 - inputs, trX1 - target, epochs - # of epochs to train the model: termined using the validation below
# batch_size verbose 2 shows at which epoch we're at

# Make predictions
trainPredict = model.predict(trainingX0)
testPredict = model.predict(testingX0)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)  # to get meters again and not normalized values
trainY = scaler.inverse_transform([trX1])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([teX1])

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# Evaluate the model: HOW TO CHOOSE NUMBER OF EPOCHS
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

# _, training_acc = model.evaluate(trX0, trX1, verbose=0)
# _, testing_acc = model.evaluate(teX0, teX1, verbose=0)
# print('Train: %.3f, Test: %.3f' % (training_acc, testing_acc))

# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Plots

# shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict
# plot baseline and predictions
plt.plot(time, scaler.inverse_transform(df), 'k')
plt.plot(time, trainPredictPlot)
plt.plot(time, testPredictPlot)
plt.show()
