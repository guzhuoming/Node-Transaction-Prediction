import pandas as pd
import numpy as np
import math
from math import sqrt
import os
import csv
from keras.models import Sequential, Model
from keras.layers import Lambda, dot, Activation, concatenate, Input, Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional, Layer
from keras import optimizers
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow
tensorflow.random.set_seed(2)

df = open('./address.csv')
dt = pd.read_csv(df)
address = dt['address']

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

def lstm(n_features=4,
         n_train=60,
         n_window=5,
         n_units=64,
         n_epochs=10,
         with_att=False,
         methods='gru',
         lr=0.001,
         n_gap = 1,
         feature_n = 1
         ):
    """

    :param n_features: 4 or 10, using 4 features or 10 features
    :param n_train: training timesteps
    :param n_window: width of training window, for example, [0 1 2 3 4]->[5], n_window = 5
    :param n_units: LSTM units
    :param n_epochs: trainning epochs
    :param feature_n: the feature_n th feature, 1 for tran_sum, 2 for tran_mean
    :return:
    """
    data = []

    for i in range(len(address)):
        f = open('./data/feature_{}_{}/{}_ft.csv'.format(n_features, n_gap, address[i]))
        df = pd.read_csv(f)
        data.append(df.values)

    data = np.array(data)
    print('data: {}, \ndata.shape(): {}'.format(data, data.shape))

    # define train, test
    scaler = MinMaxScaler(feature_range=(0, 1))
    n_samples, n_timesteps, n_features = data.shape
    scaled_data = data.reshape((n_samples, n_timesteps*n_features))
    scaled_data = scaler.fit_transform(scaled_data)
    scaled_data = scaled_data.reshape((n_samples, n_timesteps, n_features))

    # define problem properties
    n_test = n_timesteps - n_train

    # define LSTM
    # sequential
    # model = Sequential()
    # model.add(Bidirectional(LSTM(n_units, input_shape=(n_window, n_features))))
    # model.add(Dense(1))
    #
    # model.compile(loss='mse', optimizer='adam')

    # Model
    inputs = Input(shape=(n_window, n_features))
    return_sequences = False
    if with_att==True:
        return_sequences = True
    if methods=='lstm':
        att_in = Bidirectional(LSTM(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='gru':
        att_in = Bidirectional(GRU(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='rnn':
        att_in = Bidirectional(SimpleRNN(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    if with_att==True:
        att_out = attention()(att_in)
        outputs = Dense(1)(att_out)
    else:
        outputs = Dense(1)(att_in)

    model = Model(inputs, outputs)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='mse', optimizer=opt)

    # fit network
    for i in range(n_train-n_window):
        history = model.fit(scaled_data[:, i: i+n_window, :], scaled_data[:, i+n_window, feature_n], epochs=n_epochs)
        # plot history
        # plt.plot(history.history['loss'])
        # plt.show()
    # make prediction
    inv_yhat = []
    for i in range(n_test):
        yhat = model.predict(scaled_data[:, n_train-n_window+i:n_train+i, :])
        inv_yhat.append(yhat)

    inv_yhat = np.array(inv_yhat)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(3, 736, 1)
    inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], inv_yhat.shape[1]))
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(3, 736)
    inv_yhat = inv_yhat.T
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(736, 3)

    # print('scaled_data.shape={}'.format(scaled_data[:, n_train:, 0].shape))
    # inv_yhat = np.concatenate((scaled_data[:, n_train:, 0], inv_yhat), axis=1)
    # inv_yhat = inv_yhat.reshape((n_samples, n_test, 2))
    # print('inv_yhat.shape1:{}'.format(inv_yhat.shape))
    # print(inv_yhat)
    # inv_yhat = np.concatenate((inv_yhat, scaled_data[:, n_train:, 2:]), axis=2)
    # print('inv_yhat.shape2:{}'.format(inv_yhat.shape))
    # print(inv_yhat)
    # inv_yhat = np.concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)

    temp = scaled_data[:, n_train:, 0]
    temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
    for k in range(1, feature_n):
        temp_ = scaled_data[:, n_train:, k]
        temp_ = temp_.reshape(temp_.shape[0], temp_.shape[1], 1)
        temp = np.concatenate((temp, temp_), axis=2)
    inv_yhat = inv_yhat.reshape(inv_yhat.shape[0], inv_yhat.shape[1], 1)
    inv_yhat = np.concatenate((temp, inv_yhat), axis=2)
    for k in range(feature_n+1, n_features):
        temp_ = scaled_data[:, n_train:, k]
        temp_ = temp_.reshape(temp_.shape[0], temp_.shape[1], 1)
        inv_yhat = np.concatenate((inv_yhat, temp_), axis=2)
    print('inv_yhat.shape1:{}'.format(inv_yhat.shape))
    print(inv_yhat)
    inv_yhat = inv_yhat.reshape((n_samples, n_test, n_features))
    print('inv_yhat.shape2:{}'.format(inv_yhat.shape))
    print(inv_yhat)
    inv_yhat = np.concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)
    print('hhhhh={}'.format(inv_yhat.shape))

    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps*n_features)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps, n_features)
    inv_yhat[inv_yhat<0] = 0 # transform negative values to zero
    prediction = inv_yhat[:, -n_test:, feature_n]
    prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], 1)
    original = data[:, -n_test:, feature_n]
    original = original.reshape(original.shape[0], original.shape[1], 1)
    concat = np.concatenate((original, prediction), axis=2)
    print('concat.shape:{}'.format(concat.shape))
    np.set_printoptions(threshold=1e6)
    print('concat\n{}'.format(concat))
    concat = concat.reshape(concat.shape[0]*concat.shape[1], concat.shape[2])
    df = pd.DataFrame(concat)
    df.columns = ['original', 'prediction']
    if not os.path.exists('./data/LSTM_{}_{}'.format(n_features, n_gap)):
        os.makedirs('./data/LSTM_{}_{}'.format(n_features, n_gap))
    df.to_csv('./data/LSTM_{}_{}/prediction_LSTM.csv'.format(n_features, n_gap), index=False)
    rmse = sqrt(mean_squared_error(df['original'].values, df['prediction'].values))
    mae = mean_absolute_error(df['original'].values, df['prediction'].values)
    mape = mean_absolute_percentage_error(df['original'].values, df['prediction'].values)
    r2 = r2_score(df['original'].values, df['prediction'].values)
    print('rmse: {}'.format(rmse))
    print('mae: {}'.format(mae))
    print('mape: {}'.format(mape))
    print('r2: {}'.format(r2))
    return rmse

def la_ha(n_train=60,
          n_features=4,
          n_gap=1,
          n_timestamp=80
          ):

    for i in range(len(address)):
        if not os.path.exists('./data/LA_HA_{}_{}'.format(n_features, n_gap)):
            os.makedirs('./data/LA_HA_{}_{}'.format(n_features, n_gap))
        f = open('./data/LA_HA_{}_{}/{}_LA_HA.csv'.format(n_features, n_gap, address[i]), 'w', newline='')
        csvwriter = csv.writer(f)
        csvwriter.writerow(['t', 'tran_sum_real', 'tran_sum_la', 'tran_sum_ha', 'difference_la', 'difference_ha'])
        for j in range(n_timestamp-n_train):
            csvwriter.writerow([j+n_train, 0., 0., 0., 0., 0.])
        f.close()
    real_transum = np.array([])
    predict_transum_ha = np.array([])
    predict_transum_la = np.array([])

    for i in range(len(address)):
        f1 = open('./data/feature_{}_{}/{}_ft.csv'.format(n_features, n_gap, address[i]))
        df_node_pair = pd.read_csv(f1)
        f1.close()

        f2 = open('./data/LA_HA_{}_{}/{}_LA_HA.csv'.format(n_features, n_gap, address[i]))
        df_prediction = pd.read_csv(f2)
        f2.close()

        last_value = 0.
        historical_sum = 0.
        tran_sum_acc = 0

        for j in range(n_train):
            tran_sum_acc = tran_sum_acc + df_node_pair['tran_sum'][j]

        last_value = df_node_pair['tran_sum'][n_train-1]
        historical_sum = tran_sum_acc/n_train

        for j in range(n_train, n_timestamp):
            tran_sum = df_node_pair['tran_sum'][j]
            df_prediction['tran_sum_real'][j - n_train] = df_node_pair['tran_sum'][j]
            df_prediction['tran_sum_ha'][j - n_train] = tran_sum_acc/j
            tran_sum_acc = tran_sum_acc + tran_sum
            # historical_sum = historical_sum + df_prediction['tran_sum_ha'][j - n_train]
            # df_prediction['tran_sum_la'][j - n_train] = df_node_pair['tran_sum'][j-1]
            df_prediction['tran_sum_la'][j - n_train] = last_value
            df_prediction['difference_ha'][j - n_train] = df_prediction['tran_sum_ha'][j - n_train] - \
                                                    df_prediction['tran_sum_real'][j - n_train]
            df_prediction['difference_la'][j - n_train] = df_prediction['tran_sum_la'][j - n_train] - \
                                                    df_prediction['tran_sum_real'][j - n_train]
        real_transum = np.append(real_transum, df_prediction['tran_sum_real'].values)
        predict_transum_ha = np.append(predict_transum_ha, df_prediction['tran_sum_ha'].values)
        predict_transum_la = np.append(predict_transum_la, df_prediction['tran_sum_la'].values)
        df_prediction.to_csv('./data/LA_HA_{}_{}/{}_LA_HA.csv'.format(n_features, n_gap, address[i]),index=False)

    rmse_ha = mean_squared_error(real_transum, predict_transum_ha, squared=False)
    mae_ha = mean_absolute_error(real_transum, predict_transum_ha)
    mape_ha = mean_absolute_percentage_error(real_transum, predict_transum_ha)
    r2_ha = r2_score(real_transum, predict_transum_ha)

    rmse_la = mean_squared_error(real_transum, predict_transum_la, squared=False)
    mae_la = mean_absolute_error(real_transum, predict_transum_la)
    mape_la = mean_absolute_percentage_error(real_transum, predict_transum_la)
    r2_la = r2_score(real_transum, predict_transum_la)
    print('rmse_ha:{}, rmse_la:{}'.format(rmse_ha, rmse_la))
    print('mae_ha:{}, mae_la:{}'.format(mae_ha, mae_la))
    print('mape_ha:{}, mape_la:{}'.format(mape_ha, mape_la))
    print('r2_ha:{}, r2_la:{}'.format(r2_ha, r2_la))
    return rmse_ha, rmse_la, mae_ha, mae_la, mape_ha, mape_la, r2_ha, r2_la

def arima(n_train=60, p=2, d=1, q=2):
    mse = 0
    mae = 0
    mape = 0
    r_2_score = 0
    error = 0
    for i in range(len(address)):
        print(i)
        f = open('./data/feature_4_1/{}_ft.csv'.format(address[i]))
        df = pd.read_csv(f)

        data = df['tran_sum'].values
        train = data[0:n_train]
        history = [x for x in train]
        test = data[n_train:]
        pred = []
        try:
            for t in range(len(test)):
                model = ARIMA(history, order=(p,d,q))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                if yhat[0]<0:
                    yhat[0] = 0
                pred.append(yhat[0])
                history.append(test[t])
            print('pred_>=0:{}'.format(pred))
            mse = mse + mean_squared_error(test, pred)
            mae = mae + mean_absolute_error(test, pred)
            mape = mape + mean_absolute_percentage_error(test, pred)
            r_2_score = r_2_score + r2_score(test, pred)
            print(mse)
        except:
                error = error+1
                continue

    rmse = np.sqrt(mse/(len(address)-error))
    mae = mae/(len(address)-error)
    mape = mape / (len(address) - error)
    r_2_score = r_2_score / (len(address) - error)
    print('errornum:{}'.format(error))
    print('arima, rmse: {}'.format(rmse))
    print('arima, mae: {}'.format(mae))
    print('arima, mape: {}'.format(mape))
    print('arima, r_2_score: {}'.format(r_2_score))
    return rmse

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[0]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    print('agg = {}'.format(agg))
    return agg.values
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, methods='randomforest'):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        if methods=='randomforest':
            yhat = random_forest_forecast(history, testX)
        elif methods=='xgboost':
            yhat = xgboost_forecast(history, testX)
        if yhat<0:
            yhat=0
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    mse = mean_squared_error(test[:, -1], predictions)
    mae = mean_absolute_error(test[:, -1], predictions)
    mape = mean_absolute_percentage_error(test[:, -1], predictions)
    r_2_score = r2_score(test[:, -1], predictions)
    return mse, mae, mape, r_2_score, test[:, -1], predictions
def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def randomforest(n_test=20, n_features=4, n_gap=1):
    mse = 0
    mae = 0
    mape = 0
    r_2_score = 0
    for i in range(len(address)):
        print(i)
        f = open('./data/feature_4_1/{}_ft.csv'.format(address[i]))
        df = pd.read_csv(f)

        # load the dataset
        values = df['tran_sum'].values
        # transform the time series data into supervised learning
        data = series_to_supervised(values, n_in=6)
        # evaluate
        mse_, mae_, mape_, r_2_score_, y, yhat = walk_forward_validation(data, n_test, methods='randomforest')
        mse = mse+mse_
        mae = mae+mae_
        mape = mape+mape_
        r_2_score = r_2_score+r_2_score_
        # save predictions
        data2save = {}
        data2save['original'] = y
        data2save['prediction'] = yhat
        if not os.path.exists('./data/randomforest_{}_{}'.format(n_features, n_gap)):
            os.makedirs('./data/randomforest_{}_{}'.format(n_features, n_gap))
        df2save = pd.DataFrame(data2save)
        df2save.to_csv('./data/randomforest_{}_{}/{}_randomforest.csv'.format(n_features, n_gap, address[i]), index=False)
    rmse = np.sqrt(mse / len(address))
    mae = mae/len(address)
    mape = mape/len(address)
    r_2_score = r_2_score/len(address)
    print('randomforest, rmse: {}'.format(rmse))
    print('randomforest, mae: {}'.format(mae))
    print('randomforest, mape: {}'.format(mape))
    print('randomforest, r_2_score: {}'.format(r_2_score))

def xgboost(n_test=20, n_features=4, n_gap=1):
    mse = 0
    mae = 0
    mape = 0
    r_2_score = 0
    for i in range(len(address)):
        print(i)
        f = open('./data/feature_4_1/{}_ft.csv'.format(address[i]))
        df = pd.read_csv(f)

        # load the dataset
        values = df['tran_sum'].values
        # transform the time series data into supervised learning
        data = series_to_supervised(values, n_in=6)
        # evaluate
        mse_, mae_, mape_, r_2_score_, y, yhat = walk_forward_validation(data, n_test, methods='xgboost')
        mse = mse + mse_
        mae = mae + mae_
        mape = mape + mape_
        r_2_score = r_2_score + r_2_score_
        # save predictions
        data2save = {}
        data2save['original'] = y
        data2save['prediction'] = yhat

        df2save = pd.DataFrame(data2save)
        if not os.path.exists('./data/xgboost_{}_{}'.format(n_features, n_gap)):
            os.makedirs('./data/xgboost_{}_{}'.format(n_features, n_gap))
        df2save.to_csv('./data/xgboost_{}_{}/{}_xgboost.csv'.format(n_features, n_gap, address[i]), index=False)
    rmse = np.sqrt(mse / len(address))
    mae = mae / len(address)
    mape = mape / len(address)
    r_2_score = r_2_score / len(address)
    print('xgboost, rmse: {}'.format(rmse))
    print('xgboost, mae: {}'.format(mae))
    print('xgboost, mape: {}'.format(mape))
    print('xgboost, r_2_score: {}'.format(r_2_score))

def plot_curve(n_gap=1, n_features=4, n_train=60, n_timestamp=80):
    """
    plot curve of different methods
    :return:
    """
    n_test = n_timestamp-n_train

    for i in range(len(address)):
        file1 = open('./data/feature_{}_{}/{}_ft.csv'.format(n_features, n_gap, address[i]))
        df1 = pd.read_csv(file1)
        original = df1['tran_sum'].values.tolist()

        x = range(len(original))

        file2 = open('./data/LA_HA_{}_{}/{}_LA_HA.csv'.format(n_features, n_gap, address[i]))
        df2 = pd.read_csv(file2)
        la = df2['tran_sum_la'].values.tolist()
        ha = df2['tran_sum_ha'].values.tolist()
        # la = original[0:n_train]+la
        # ha = original[0:n_train]+ha

        file3 = open('./data/xgboost_{}_{}/{}_xgboost.csv'.format(n_features, n_gap, address[i]))
        df3 = pd.read_csv(file3)
        xgboost_ = df3['prediction'].values.tolist()

        file4 = open('./data/randomforest_{}_{}/{}_randomforest.csv'.format(n_features, n_gap, address[i]))
        df4 = pd.read_csv(file4)
        randomforest_ = df4['prediction'].values.tolist()

        file5 = open('./data/LSTM_{}_{}/prediction_LSTM.csv'.format(n_features, n_gap, address[i]))
        df5 = pd.read_csv(file5)
        lstm_ = df5['prediction'].values.tolist()
        lstm_ = lstm_[i*n_test: i*n_test+n_test]
        # lstm_ = original[0:n_train]+lstm_

        plt.figure()
        plt.plot(x, original)
        # plt.plot(range(n_train, n_timestamp), la)
        # plt.plot(range(n_train, n_timestamp), ha)
        plt.plot(range(n_train, n_timestamp), xgboost_)
        plt.plot(range(n_train, n_timestamp), randomforest_)
        plt.plot(range(n_train, n_timestamp), lstm_)
        plt.xlabel('time')
        plt.ylabel('transaction value')
        # plt.legend(('original', 'LA', 'HA', 'xgboost', 'randomforest', 'GRU'))
        plt.legend(('original', 'xgboost', 'randomforest', 'GRU'))
        plt.title(address[i])
        plt.show()

if __name__=='__main__':
    # rmse_li = []
    # for n_train in range(50, 75):
    #     rmse = lstm(n_features=4, n_train=n_train, n_window=10, n_units=100, n_epochs=10, n_gap=1, with_att=False, methods="gru", feature_n=1)
    #     rmse_li.append(rmse)
    # plt.plot(range(50,75), rmse_li)
    # plt.title("rmse")
    # plt.xlabel("n_train")
    # plt.ylabel("rmse")
    # plt.show()

    # rmse_lstm = lstm(n_features=4, n_train=60, n_window=10, n_units=100, n_epochs=10, n_gap=1, with_att=True, methods="gru", feature_n=1)
    # rmse_ha, rmse_la = la_ha()
    # xgboost()
    # randomforest()

    # arima_rmse = []
    # for p in range(1,4):
    #     for d in range(1,2):
    #         for q in range(1,4):
    #             rmse = arima(p=p,d=d,q=q)
    #             s = 'pqd, p={}, d={}, q={}, rmse={}'.format(p,d,q,rmse)
    #             arima_rmse.append(s)
    # for i in range(len(arima_rmse)):
    #     print(arima_rmse[i])


    # plot_curve()
    # arima(p=1,d=0,q=0)
    # xgboost()
    # randomforest()
    # la_ha()
    lstm()