# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from pandas import DataFrame
from pandas import concat
from numpy import argmax
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from keras.models import load_model
from tools import statistics
import math

# convert time series into supervised learning problem
# data: Sequence of observations as a list or 2D NumPy array. Required.
# n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
# n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
# dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg = agg.applymap(lambda x: np.int32(x))
    return agg

# convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = 3
    Xstr = []
    for pattern in X:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        Xstr.append(element_ensem)
    ystr=[]
    for pattern in y:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        ystr.append(element_ensem)
    return Xstr,ystr

def one_hot_encode(X, series_min,series_max,n_unique):
    gap=(series_max-series_min)/n_unique
    Xenc=[]
    for sequence in X:
        new_index_ensem=[]
        for value in sequence:
            new_index=(value-series_min)/gap
            if value == 18544:
                new_index = new_index-0.1
            new_index_ensem.append(int(new_index))
        encoding=[]
        if value == 18544:
            print(new_index_ensem, new_index, value, series_max, series_min, gap)
        for index in new_index_ensem:
            vector=[0 for _ in range(n_unique)]
            vector[index]=1
            encoding.append(vector)
        Xenc.append(encoding)
    return np.array(Xenc)

# decode a one hot encoded string
def one_hot_decode(y,series_min,series_max,n_unique):
    gap=(series_max-series_min)/n_unique
    y_dec=[]
    for encoded_seq in y:
        decoded_seq=[argmax(vector) for vector in encoded_seq]
        decoded_seq=np.array(decoded_seq)
        decoded_seq_tran=list(decoded_seq*gap+series_min)
        y_dec.append(decoded_seq_tran)
    return y_dec


def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

if __name__=='__main__':
    # load raw data
    df_raw = pd.read_csv('../data/load.csv', header=0, usecols=[0, 1])
    # numpy array
    df_raw_array = df_raw.values
    # daily load
    list_hourly_load = [df_raw_array[i, 1] / 1000 for i in range(0, len(df_raw))]
    print ("Data shape of list_hourly_load: ", np.shape(list_hourly_load))
    k = 0
    for j in range(0, len(list_hourly_load)):
        if (abs(list_hourly_load[j] - list_hourly_load[j - 1]) > 2 and abs(
                    list_hourly_load[j] - list_hourly_load[j + 1]) > 2):
            k = k + 1
            list_hourly_load[j] = (list_hourly_load[j - 1] + list_hourly_load[j + 1]) / 2 + list_hourly_load[j - 24] - \
                                  list_hourly_load[j - 24 - 1] / 2
        sum = 0
        num = 0
        for t in range(1, 8):
            if (j - 24 * t >= 0):
                num = num + 1
                sum = sum + list_hourly_load[j - 24 * t]
            if (j + 24 * t < len(list_hourly_load)):
                num = num + 1
                sum = sum + list_hourly_load[j + 24 * t]
        sum = sum / num
        if (abs(list_hourly_load[j] - sum) > 3):
            k = k + 1
            if (list_hourly_load[j] > sum):
                list_hourly_load[j] = sum + 3
            else:
                list_hourly_load[j] = sum - 3
    print(k)
    # plt.plot(list_hourly_load)
    # plt.show()
    # shift all data by mean
    list_hourly_load = np.array(list_hourly_load)
    shifted_value = list_hourly_load.mean()
    list_hourly_load -= shifted_value
    # the length of the sequnce for predicting the future value
    sequence_length = 25
    # convert the vector to a 2D matrix
    matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)
    matrix_load = np.array(matrix_load)
    print ("Data shape: ", matrix_load.shape)
    # train_row = int(round(0.9 * matrix_load.shape[0]))
    train_row = matrix_load.shape[0] - 24*14
    print('train:', train_row, 'test:', 24*14)
    train_set = matrix_load[:train_row, :]
    # random seed
    np.random.seed(1234)
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(train_set)
    # the training set
    X_train = train_set[:, :-1]
    # the last column is the true value to compute the mean-squared-error loss
    y_train = train_set[:, -1]
    # print(X_train[0], y_train[0])
    # the test set
    X_test = matrix_load[train_row:, :-1]
    y_test = matrix_load[train_row:, -1]
    time_test = [df_raw_array[i, 0] for i in range(train_row + 23, len(df_raw))]
    # print(time_test[0])
    # the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(np.shape(X_train), np.shape(y_train))
    # create LSTM
    model = Sequential()
    model.add(LSTM(150, batch_input_shape=(None,X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(RepeatVector(1))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(1, activation='linear')))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    # show model
    # print(model.summary())
    # train LSTM
    history=model.fit(X_train, y_train, epochs=50, batch_size=50, validation_split=0.05, shuffle=False, verbose=2)
    # save model
    model.save('../seq2seq.h5')
    # model = load_model('../model/seq2seq.h5')
    # evaluate on some new patterns
    predicted_values = model.predict(X_test)
    num_test_samples = len(predicted_values)
    predicted_values = np.reshape(predicted_values, (num_test_samples, 1))
    # evaluation
    mape = statistics.mape((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MAPE is ', mape)
    mae = statistics.mae((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MAE is ', mae)
    mse = statistics.meanSquareError((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MSE is ', mse)
    rmse = math.sqrt(mse)
    print('RMSE is ', rmse)
    nrmse = statistics.normRmse((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('NRMSE is ', nrmse)
    # plot the results
    fig = plt.figure()
    plt.plot(y_test + shifted_value, label="$Observed$", c='green')
    plt.plot(predicted_values + shifted_value, label="$Predicted$", c='red')
    plt.xlabel('Hour')
    plt.ylabel('Electricity load, kW')
    plt.legend()
    plt.show()
