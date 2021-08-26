from typing import Dict, Callable, Any
import numpy as np
import pandas as pd

# Univariate Statistical Models
from statsmodels.tsa.ar_model import AutoReg    # Auto Regression
from statsmodels.tsa.arima_model import ARMA    # Auto Regressive Moving Average
from statsmodels.tsa.arima.model import ARIMA    # Auto Regressive Integrated Moving Average
from statsmodels.tsa.holtwinters import SimpleExpSmoothing    # Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing    # Holt-winters Exponential Smoothing

from statsmodels.tsa.vector_ar.var_model import VAR    # Vector Auto regression
from statsmodels.tsa.statespace.varmax import VARMAX    # Vector Autoregressive Moving Average with
                                                        # eXogenous regressors model
from statsmodels.tsa.statespace.sarimax import SARIMAX    # Seasonal Auto regressive integrated moving average

# Tensorflow library
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Lambda, Conv1D, Reshape
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Model Decorator
# def stats_method(func: Callable) -> Callable[[object, list, tuple, int], Any]:
#     def wrapper(obj: object, history: list, cfg: tuple, pred_step: int):
#         return func(obj, history, cfg, pred_step)
#     return wrapper


class Algorithm(object):
    """
    Statistical Model Class

    # Model List
    1. Uni-variate Model
        - AR model (Autoregressive model)
        - ARMA model (Autoregressive moving average model)
        - ARIMA model (Autoregressive integrated moving average model)
        - SES model (Simple Exponential Smoothing model)
        - HWES model (Holt Winters Exponential Smoothing model)

    2. Multivariate Model
        - VAR model (Vector Autoregressive model)
        - VARMA model (Vector Autoregressive Moving Average model)
        - VARMAX model (Vector Autoregressive Moving Average with eXogenous regressors model)
    """

    #############################
    # Uni-variate Model
    #############################
    # Auto-regressive Model
    @staticmethod
    def ar(history, cfg: tuple, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                l: lags (int)
                t: trend ('c', 'nc')
                    c: Constant
                    nc: No Constant
                s: seasonal (bool)
                p: period (Only used if seasonal is True)
        :param pred_step: prediction steps
        :return: forecast result
        """
        lag, t, s, p = cfg
        model = AutoReg(endog=history, lags=lag, trend=t, seasonal=s, period=p)
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step prediction
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    # Auto regressive moving average model
    @staticmethod
    def arma(history, cfg: tuple, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                f: frequency of the time series (‘B’, ‘D’, ‘W’, ‘M’, ‘A’, ‘Q)
                t: trend ('c', 'nc')
                    c: Constant
                    nc: No constant
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = cfg
        # define model
        model = ARMA(endog=history, order=o, freq=f)
        # fit model
        model_fit = model.fit(trend=t, disp=0)
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    # Autoregressive integrated moving average model
    @staticmethod
    def arima(history, cfg: tuple, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                o: order (p, d, q)
                    p: Trend autoregression order
                    d; Trend difference order
                    q: Trend moving average order
                f: frequency of the time series (‘B’, ‘D’, ‘W’, ‘M’, ‘A’, ‘Q)
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = cfg
        # define model
        model = ARIMA(history, order=o, trend=t, freq=f)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @staticmethod
    def ses(history, cfg: tuple, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                i: initialization_method (None, 'estimated', 'heuristic', 'legacy-heuristic', 'known')
                    - Method for initialize the recursions
                l: smoothing level (float)
                o: optimized or not (bool)
        :param pred_step: prediction steps
        :return: forecast result
        """
        i, s, o = cfg
        # define model
        model = SimpleExpSmoothing(history, initialization_method=i)
        # fit model
        model_fit = model.fit(smoothing_level=s, optimized=o)  # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @staticmethod
    def hw(history, cfg: tuple, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                t: trend ('add', 'mul', 'additive', 'multiplicative')
                    - type of trend component
                d: damped_trend (bool)
                    - should the trend component be damped
                s: seasonal ('add', 'mul', 'additive', 'multiplicative', None)
                    - Type of seasonal component
                p: seasonal_periods (int)
                    - The number of periods in a complete seasonal cycle
                b: use_boxcox (True, False, ‘log’, float)
                    - Should the Box-Cox transform be applied to the data first?
                r: remove_bias (bool)
                    - Remove bias from forecast values and fitted values by enforcing that the average residual is
                      equal to zero
        :param pred_step: prediction steps
        :return: forecast result
        """
        t, d, s, p, b, r = cfg
        # define model
        model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s,
                                     seasonal_periods=p, use_boxcox=b)
        # fit model
        model_fit = model.fit(optimized=True, remove_bias=r)  # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    # def prophet(self, history, n_test, forecast=False):
    #
    #     data = history.reset_index(level=0)
    #     data = data.rename(columns={'dt': 'ds', 'sales': 'y'})
    #
    #     model = Prophet()
    #     if not forecast:
    #         train = data.iloc[:-n_test, :]
    #         test = data.iloc[-n_test:, :]
    #
    #         model.fit(train)
    #         future = model.make_future_dataframe(periods=n_test)
    #         forecast = model.predict(future)
    #
    #         error = self.calc_sqrt_mse(test['y'], forecast['yhat'][-n_test:])
    #
    #         return error
    #
    #     else:
    #         model.fit(history)
    #         future = model.make_future_dataframe(periods=n_test)
    #         forecast = model.predict(future)
    #
    #         return  forecast['yhat'][-n_test:]

    #############################
    # Multi-variate Model
    #############################
    @staticmethod
    def var(history: np.array, cfg, pred_step=0):
        """
        :param history:
        :param cfg:
        :param pred_step:
        :return:
        """
        lag = cfg
        # define model
        model = VAR(history)
        # fit model
        model_fit = model.fit(lag)
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        # yhat = model.predict(start=len(history), end=len(history) + pred_step - 1)
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    # Vector Autoregressive Moving Average model
    @staticmethod
    def varma(history: np.array, cfg, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, t = cfg
        # define model
        model = VARMAX(history.astype(float), order=o, trend=t)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    # Vector Autoregressive Moving Average with eXogenous regressors model
    @staticmethod
    def varmax(history: np.array, data_exog: np.array, cfg, pred_step=0):
        """
        :param history: time series data
        :param data_exog: exogenous data
        :param cfg:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, t = cfg
        # define model
        model = VARMAX(history, exog=data_exog, order=o, trend=t)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    def seq2seq(self) -> Model:
        """
        Sequence to sequence model
        # Multivariate to target
        # m time-steps to n time-steps
        :return:
        """
        # define sequence to sequence LSTM model
        input_train = Input(shape=(self.window_input, self.n_features))
        output_train = Input(shape=(self.window_output, self.n_features))

        # Encoder LSTM
        encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
            self.n_hidden, activation=self.activation,
            dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
            return_sequences=False, return_state=True)(input_train)

        # Batch normalization
        encoder_last_h1 = BatchNormalization(momentum=self.momentum)(encoder_last_h1)
        encoder_last_c = BatchNormalization(momentum=self.momentum)(encoder_last_c)

        # Decoder LSTM
        decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
        decoder = LSTM(self.n_hidden, activation=self.activation, dropout=self.dropout,
                       recurrent_dropout=self.recurrent_dropout, return_state=False,
                       return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
        out = TimeDistributed(Dense(1))(decoder)

        model = Model(inputs=input_train, outputs=out)

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mse'])
        model.summary()

        return model


    # @staticmethod
    # def prophet(history, n_test, forecast=False):
    #
    #     data = history.reset_index(level=0)
    #     data = data.rename(columns={'dt': 'ds', 'sales': 'y'})
    #
    #     model = Prophet()
    #     if not forecast:
    #         train = data.iloc[:-n_test, :]
    #         test = data.iloc[-n_test:, :]
    #
    #         model.fit(train)
    #         future = model.make_future_dataframe(periods=n_test)
    #         forecast = model.predict(future)
    #
    #         error = self.calc_sqrt_mse(test['y'], forecast['yhat'][-n_test:])
    #
    #         return error
    #
    #     else:
    #         model.fit(history)
    #         future = model.make_future_dataframe(periods=n_test)
    #         forecast = model.predict(future)
    #
    #         return  forecast['yhat'][-n_test:]

    # def lstm_train(self, train: pd.DataFrame, units: int) -> float:
    #     # scaling
    #     scaler = MinMaxScaler()
    #     train_scaled = scaler.fit_transform(train)
    #     train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    #
    #     x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
    #                                                n_steps_out=self.n_test)
    #
    #     n_features = x_train.shape[2]
    #
    #     # Build model
    #     model = Sequential()
    #     model.add(LSTM(units=units, activation='relu', return_sequences=True,
    #               input_shape=(self.n_test, n_features)))
    #     # model.add(LSTM(units=units, activation='relu'))
    #     model.add(Dense(n_features))
    #     model.compile(optimizer='adam', loss=self.root_mean_squared_error)
    #
    #     history = model.fit(x_train, y_train,
    #                         epochs=config.EPOCHS,
    #                         batch_size=config.BATCH_SIZE,
    #                         validation_split=1-config.TRAIN_RATE,
    #                         shuffle=False,
    #                         verbose=0)
    #     self.epoch_best = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    #
    #     rmse = min(history.history['val_loss'])
    #
    #     return rmse
    #
    # def lstm_predict(self, train: pd.DataFrame, units: int) -> np.array:
    #     # scaling
    #     scaler = MinMaxScaler()
    #     train_scaled = scaler.fit_transform(train)
    #     train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    #
    #     x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
    #                                                n_steps_out=self.n_test)
    #     n_features = x_train.shape[2]
    #
    #     # Build model
    #     model = Sequential()
    #     model.add(LSTM(units=units, activation='relu', return_sequences=True,
    #               input_shape=(self.n_test, n_features)))
    #     # model.add(LSTM(units=units, activation='relu'))
    #     model.add(Dense(n_features))
    #     model.compile(optimizer='adam', loss=self.root_mean_squared_error)
    #
    #     model.fit(x_train, y_train,
    #               epochs=self.epoch_best,
    #               batch_size=config.BATCH_SIZE,
    #               shuffle=False,
    #               verbose=0)
    #     test = x_train[-1]
    #     test = test.reshape(1, test.shape[0], test.shape[1])
    #
    #     predictions = model.predict(test, verbose=0)
    #     predictions = scaler.inverse_transform(predictions[0])
    #
    #     return predictions[:, 0]

    @staticmethod
    def lstm_data_reshape(data: np.array, n_feature: int) -> np.array:
        return data.reshape((data.shape[0], data.shape[1], n_feature))

    # @staticmethod
    # def root_mean_squared_error(y_true, y_pred):
    #     return K.sqrt(K.mean(K.square(y_pred - y_true)))
