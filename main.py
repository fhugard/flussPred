__author__ = "Florence Hugard, Malik Lechekhab"
__copyright__ = "Copyright 2019, Fluss Project"
__credits__ = ["Felan Carlo Garcia", "Jason Brownlee"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Malik Lechekhab"
__email__ = "malik.lechekhab@unil.ch"
__status__ = "Development"

# Importations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import math
import datetime
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator


def create_dataset(dataset, sliding_window, look_fwd):
    """Create a matrix of shifted inputs and associated observation for a given
       sliding windows and forcast delay
       e.g. sliding_window = 30, look_fwd = 7:
        X = [t0  t1  t2  ...]
            [t1  t2  t3  ...]
                     .
            [t30 t31 t32 ...]
            _________________
        Y = [t37 t38 t39 ...]

    Attributes:
        dataset (dataframe:float): full dataset to shift.
        slidning_window (int): sliding window from which the number of row of
                               the matrix is based.
        look_fwd(int): additional shift corresponding to the prediction delay,
                       e.g. look_fwd = 7 => prediction in 1 week.

    Return:
        dataX(array:float): Matrix X.
        dataY(array:float): Vector Y.

    Todo:
        * Parallelize the process
    """

    dataX, dataY = [], []

    # Shift the values
    for i in range(len(dataset) - sliding_window - look_fwd - 1):
        # First shift (sliding window)
        a = dataset[i:(i+sliding_window), 0]
        dataX.append(a)
        # Additional shift (prediction delay)
        dataY.append(dataset[i + sliding_window + look_fwd, 0])

    return np.array(dataX), np.array(dataY)


def rmse(observations, predictions):
    """Compute the Root Mean Squaed Error: (E[(Yt - Yt_hat)^2])^(0.5)

    Attributes:
        observations (dataframe:float): observations (Y)
        predictions (dataframe:float): prediction (Y_hat)

    Return:
        score (float): Value of the Root Mean Squared Error

    Todo:
        * Parallelize the process
    """
    errors = pow(observations - predictions, 2)
    score = math.sqrt(errors.mean())

    return score


def lstm(s_window, l_fwd, n_units, n_epochs, batch_size):
    """Compute the Root Mean Squaed Error: (E[(Yt - Yt_hat)^2])^(0.5)

    Attributes:
        s_window (int): Sliding window hyperparameter
        l_fwd (int): Prediction delay hyperparameter
        n_units (int): Number of units hyperparameter
        n_epochs (int): Number of epochs hyperparameter
        batch_size (int): Size of the batch hyperparameter

    Todo:
        * Create to function: one for creating the model, one for plotting
    """

    slide_window = s_window
    look_fwd = l_fwd

    # Print the hyperparameters
    print('LSTM\nHyperparameters: slide window: {}, look fwd: {}, units: {}, epochs: {}, batch size: {}'.format(s_window, l_fwd, n_units, n_epochs, batch_size))

    # Define the plot styles
    plt.style.use('ggplot')

    # Fix random seed for reproducibility
    np.random.seed(10)

    # Use the elevation_PRES.csv dataset
    df = pd.read_csv('C:/Users/leche/OneDrive/Documents/HEC/Advanced Data Analytics/Project/PROJECT_ADA/elevation_PRES.csv', sep=';', header=None,
                     names=['Index', 'Time', 'Elevation (in meters)'])

    ## Format the dataframe
    # Drop the first column
    df = df.drop(df.columns[[0]], axis=1)

    # Reformat the dates
    df['Time'] = df['Time'].map(lambda x: str(x)[:-11])
    df['Time'] = pd.to_datetime(df['Time'])

    # Set the date as index
    time = df.iloc[:, 0].values
    df = df.set_index('Time')

    # Normalize the data for model purposes
    scaler  = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)

    # Split into train (70%) and test (30%) sets
    train_size  = int(len(dataset) * 0.7)
    test_size   = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # Shift the data
    trainX, trainY = create_dataset(train, slide_window, look_fwd)
    testX, testY   = create_dataset(test, slide_window, look_fwd)

    # Reshape the data for model purposes
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    ## Set the model
    # Select the model
    model = Sequential()

    # Add the 1st layer of units
    model.add(LSTM(units=n_units, return_sequences=True, input_shape=(trainX.shape[1], slide_window)))

    # Add the 2nd layer of units
    model.add(LSTM(units=n_units))

    # Add the dense layer
    model.add(Dense(units=1))

    # Compile the model with given loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Print when the model is ready
    print('The model is compiled')

    # Fit the model
    model1 = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=n_epochs, batch_size=batch_size, verbose=2)

    # Save the predictions
    trainPredict = model.predict(trainX)
    testPredict  = model.predict(testX)

    # Compute Root Mean Squared Error for the traning set and test set
    trainScore = rmse(trainY, trainPredict[:,0])
    testScore = rmse(testY, testPredict[:,0])

    # Denormalize the value
    dataset = scaler.inverse_transform(dataset)
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    ## Format the data for plotting
    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[slide_window:len(trainPredict)+slide_window, :] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (slide_window*2 + look_fwd*2) + 1 : len(dataset) - 1, :] = testPredict

    # Set the danger levels
    dl2 = 372.4
    dl3 = 372.55
    dl4 = 372.7
    dl5 = 372.95

    ## Plots
    # Plot data overview
    plt.title('Dataset overview', fontsize=10)
    plt.plot(time, dataset, label='full set', color=('#3B518B'), linewidth=0.5)
    # Add danger levels
    plt.axhspan(dl2, dl3, color=('#ffe200'), alpha=0.4)
    plt.axhspan(dl3, dl4, color=('#e9772a'), alpha=0.4)
    plt.axhspan(dl4, dl5, color=('#fa4d4d'), alpha=0.4)
    plt.axhspan(dl5, dl5 + 1, color=('#9c0202'), alpha=0.4)
    plt.text(time[-1], dl2, 'DL2', fontsize=9, color=('#b4a004'))
    plt.text(time[-1], dl3, 'DL3', fontsize=9, color=('#e9772a'))
    plt.text(time[-1], dl4, 'DL4', fontsize=9, color=('#fa4d4d'))
    plt.text(time[-1], dl5, 'DL5', fontsize=9,color=('#9c0202'))

    plt.ylabel('elevation in meter', fontsize=8)
    plt.ylim(top=(dataset.max() + 0.1))
    plt.legend()

    plt.tight_layout()

    plt.show()

    # Plot the traning process and testing process
    plt.title('LSTM\nHyperparam.: slide window: {}, look fwd: {}, units: {}, epochs: {}, batch size: {}'.format(s_window, l_fwd, n_units, n_epochs, batch_size), fontsize=10)
    # Add full data
    plt.plot(time, dataset, label='full set', ms=1, color=('#3B518B'), linewidth=0.5)
    # Add training set prediction
    plt.plot(time, trainPredictPlot, label='train predict.', ms=1, color=('#29AE80'), linewidth=0.5)
    # Add testing set prediction
    plt.plot(time, testPredictPlot, label='test predict.', ms=1, color=('#85D349'), linewidth=0.5)
    plt.xlabel('root mean squared errors\n-\ntrain: {:f} m.\ntest: {:f} m.'.format(trainScore, testScore), fontsize=8)
    plt.legend()

    plt.tight_layout()

    ## ----------UNCOMMENT FOR SAVING FILES---------
    # filename_full = 'C:/Users/leche/OneDrive/Documents/HEC/Advanced Data Analytics/Project/PROJECT_ADA/plots/plot-' + str(s_window) + '_' + str(l_fwd) + '_' + str(n_units) + '_' + str(n_epochs) + '_' + str(batch_size) + '-full.png'
    # plt.savefig(filename_full)
    # plt.cla

    ## ----------COMMENT FOR SAVING FILES---------
    plt.show()

    ## Apply the model to out of sample data
    # Use the elevation_PRES.csv dataset
    unseen = pd.read_csv('C:/Users/leche/OneDrive/Documents/HEC/Advanced Data Analytics/Project/PROJECT_ADA/elevation_FUT.csv', sep=';', header=None,
                     names=['Index', 'Time', 'Elevation (in meters)'])

    ## Format the dataframe
    # Drop the first column
    unseen = unseen.drop(unseen.columns[[0]], axis=1)

    # Reformat the dates
    unseen['Time'] = unseen['Time'].map(lambda x: str(x)[:-11])
    unseen['Time'] = pd.to_datetime(unseen['Time'])

    # Set the date as index
    time = unseen.iloc[:, 0].values
    unseen = unseen.set_index('Time')

    # Clean the data
    unseen_clean = []
    for i in range(len(unseen)):
        unseen_clean.append(unseen.iloc[i])
    unseen_clean = np.asarray(unseen_clean).astype('float32')

    # Normalize the data
    unseen_clean = scaler.fit_transform(unseen_clean)

    # Shift the data
    features, labels = create_dataset(unseen_clean, slide_window, look_fwd)
    features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

    # Run the model
    unseen_results_norm = model.predict(features)

    # Denomalize the data
    unseen_results = scaler.inverse_transform(unseen_results_norm)

    # Format the data for plotting
    unseen_results_plot = pd.DataFrame().reindex_like(unseen[slide_window+look_fwd :-1])
    unseen_results_plot['Elevation (in meters)'] = unseen_results

    # Create a naive model (e.g. "prediction" in t+x = t)
    naive = unseen
    naive = naive.shift(look_fwd, axis=0)

    # Compute the Root Mean Squared Errors of the models
    score_lstm = rmse(unseen[slide_window + look_fwd :-1], unseen_results[:])
    score_naive = rmse(naive[slide_window + look_fwd:-1], unseen[slide_window + look_fwd :-1])

    ## Plots
    # Plot the observations, lstm predictions and naive prediction
    plt.subplot(2, 2, (1, 2))
    # Plot the observations
    plt.plot(unseen, label='observed', marker='.', ms=2, color=('#3B518B'), linewidth=1)
    # Plot the naive predictions
    plt.plot(naive[slide_window + look_fwd:-1], label='naive', color=('#808080'), linewidth=1)
    # Plot the lstm predictions
    plt.plot(unseen_results_plot, label='predicted', marker='.', ms=2, color=('#85D349'), linewidth=1)
    # Show the danger levels line if the data are closed to them
    lvl_max = unseen['Elevation (in meters)'].max()
    # Show danger level 2
    if lvl_max + 0.1 > dl2:
        plt.axhline(dl2, color=('#ffe200'), alpha=0.5)
        plt.text(time[-1], dl2, 'DL2', fontsize=9, color=('#b4a004'))
    # Show danger level 3
    if lvl_max + 0.1 > dl3:
        plt.axhline(dl3, color=('#e9772a'), alpha=0.55)
        plt.text(time[-1], dl3, 'DL3', fontsize=9, color=('#e9772a'))
    # Show danger level 4
    if lvl_max + 0.1 > dl4:
        plt.axhline(dl4, color=('#fa4d4d'), alpha=0.5)
        plt.text(time[-1], dl4, 'DL4', fontsize=9, color=('#fa4d4d'))
    # Show danger level 5
    if lvl_max + 0.1 > dl5:
        plt.axhline(dl5, color=('#9c0202'), alpha=0.5)
        plt.text(time[-1], dl5, 'DL5', fontsize=9,color=('#9c0202'))
    # Show the first sliding window
    plt.axvspan(time[0], time[slide_window], color=('#F5E626'), alpha=0.25)
    # Show the first prediction delay
    plt.axvspan(time[slide_window], time[slide_window+look_fwd], color=('#29AE80'), alpha=0.25)

    plt.title('LSTM\nHyperparam.: slide window: {}, look fwd: {}, units: {}, epochs: {}, batch size: {}'.format(s_window, l_fwd, n_units, n_epochs, batch_size), fontsize=10)
    plt.xlabel('yellow: slide window, blue: look forward', fontsize=8)
    plt.ylabel('elevation in meter', fontsize=8)
    plt.tick_params(labelsize=6)
    plt.legend()

    # Plot the model losses when training
    plt.subplot(2, 2, 3)
    # Set the labels
    lab = [i for i in range(1, n_epochs + 1)]
    # Plot result of the loss function for a given epoch
    plt.plot(lab, model1.history['loss'], label='train', ms=3, color=('#29AE80'), linestyle= '--')
    # Plot the loss value
    plt.plot(lab, model1.history['val_loss'], label='test', ms=3, color=('#29AE80'), linestyle=':')
    plt.legend(loc='upper right')
    plt.title('Training model', fontsize=10)
    plt.ylabel('loss', fontsize=8)
    plt.xlabel('epochs', fontsize=8)
    plt.tick_params(labelsize=6)

    # Plot the Root Mean Squared Errors of the LSTM and Naive model
    plt.subplot(2, 2, 4)
    # Set labels
    rmse_labels = ['LSTM', 'Naive']
    # Set the bars
    rmse_values = [score_lstm, score_naive]
    # Plot the bars
    plt.bar(rmse_labels, rmse_values, color=[('#29AE80'),('#808080')], alpha=0.5)
    plt.title('Score', fontsize=10)
    plt.xlabel('root mean squared error LSTM: {:f} m\nroot mean squared error Naive: {:f} m'.format(score_lstm, score_naive), fontsize=8)
    plt.ylabel('rmse', fontsize=8)
    plt.tick_params(labelsize=6)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    ## ----------UNCOMMENT FOR SAVING FILES---------
    # filename = 'C:/Users/leche/OneDrive/Documents/HEC/Advanced Data Analytics/Project/PROJECT_ADA/plots_slide/plot-' + str(s_window) + '_' + str(l_fwd) + '_' + str(n_units) + '_' + str(n_epochs) + '_' + str(batch_size) + '.png'
    # plt.savefig(filename)
    # plt.cla

    ## ----------COMMENT FOR SAVING FILES---------
    plt.show()


os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

## Uncomment for hyper parameter tunning
# # Nb of units
# for i in [16, 32, 64]:
#     # Nb of epochs
#     for j in [20, 30]:
#         # Size of the batch
#         for k in [16, 32, 64]:
#             lstm(s_window=90, l_fwd=30, n_units=i, n_epochs=j, batch_size=k)

# Run the model
lstm(s_window=90, l_fwd=30, n_units=16, n_epochs=1, batch_size=32)
