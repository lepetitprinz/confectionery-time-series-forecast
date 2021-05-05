from DataPrep import DataPrep
import config

from typing import Dict, Callable, Any
import warnings
from math import sqrt

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

# Univariate Statistical Models
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from prophet import Prophet

warnings.filterwarnings('ignore')


# Model Decorator
def stats_method(func: Callable) -> Callable[[object, list, tuple, int], Any]:
    def wrapper(obj: object, history: list, cfg: tuple, pred_step: int):
        return func(obj, history, cfg, pred_step)
    return wrapper


class ModelStats(object):
    """
    Statistical Model Class

    # Model List
    1. Univariate Model
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

    def __init__(self, sell_in, sell_out):
        # model type
        self.model_type = config.VAR_TYPE

        # dataset
        self.df_sell = {'sell_in': sell_in,
                        'sell_out': sell_out}

        # model
        self.model_univ: Dict[str, stats_method] = {'ar': self.ar,
                                                    'arma': self.arma,
                                                    'arima': self.arima,
                                                    'ses': self.ses,
                                                    'hw': self.hw}
        # hyper-parameters
        self.cfg_ar = (config.LAG, config.TREND, config.SEASONAL, config.PERIOD)
        self.cfg_arma = (config.TWO_LVL_ORDER, config.FREQUENCY, config.TREND_ARMA)
        self.cfg_arima = (config.THR_LVL_ORDER, config.FREQUENCY, config.TREND_ARMA)
        self.cfg_ses = (config.INIT_METHOD, config.SMOOTHING, config.OPTIMIZED)
        self.cfg_hw = (config.TREND_HW, config.DAMPED_TREND, config.SEASONAL_MTD, config.PERIOD,
                       config.USE_BOXCOX, config.REMOVE_BIAS)
        self.cfg_dict = {'ar': self.cfg_ar,
                         'arma': self.cfg_arma,
                         'arima': self.cfg_arima,
                         'ses': self.cfg_ses,
                         'hw': self.cfg_hw}

    def train(self):
        if self.model_type == 'univ':
            best_models = self.df_sell.copy()
            for sell_type, sell in best_models.items():
                for group_type, group in sell.items():
                    for grp, time_range in group.items():
                        for time_type, df in time_range.items():
                            best_model = self.get_best_models(df=df)
                            time_range[time_type] = best_model

            return best_models

    def predict(self, model):
        return

    def get_best_models(self, df):
        best_models = []
        for model in config.MODEL_CANDIDATES[self.model_type]:
            score = self.walk_fwd_validation_univ(model=model, model_cfg=self.cfg_dict[model],
                                                  data=df, n_test=config.N_TEST)
            best_models.append([model, score])

        # prophet model
        score = self.prophet(history=df, n_test=config.N_TEST)
        best_models.append(['prophet', score])

        best_models = sorted(best_models, key=lambda x: x[1])

        return best_models[0]

    # Auto-regressive Model
    @stats_method
    def ar(self, history, cfg: tuple, pred_step=0):
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
        l, t, s, p = cfg
        model = AutoReg(endog=history, lags=l, trend=t, seasonal=s, period=p)
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    # Auto regressive moving average model
    @stats_method
    def arma(self, history, cfg: tuple, pred_step=0):
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
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    # Autoregressive integrated moving average model
    @stats_method
    def arima(self, history, cfg: tuple, pred_step=0):
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
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    @stats_method
    def ses(self, history, cfg: tuple, pred_step=0):
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
        model_fit = model.fit(smoothing_level=s, optimized=o)     # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    @stats_method
    def hw(self, history, cfg: tuple, pred_step=0):
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
        model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p,
                                     use_boxcox=b)
        # fit model
        model_fit = model.fit(optimized=True, remove_bias=r)     # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    def prophet(self, history, n_test):
        data = history.reset_index(level=0)
        data = data.rename(columns={'dt': 'ds', 'sales': 'y'})
        train = data.iloc[:-n_test, :]
        test = data.iloc[-n_test:, :]

        model = Prophet()
        model.fit(train)

        future = model.make_future_dataframe(periods=n_test)
        forecast = model.predict(future)

        error = self.calc_sqrt_mse(test['y'], forecast['yhat'][-n_test:])

        return error

    @staticmethod
    def train_test_split(data: pd.DataFrame, n_test):
        return data.values[:-n_test], data.values[-n_test:]

    @staticmethod
    def calc_sqrt_mse(actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    def score_model(self, model: str, data, n_test, cfg):
        # convert config to a key
        key = str(cfg)
        result = self.walk_fwd_validation_univ(model=model, data=data, n_test=n_test,
                                               model_cfg=cfg)
        return model, key, result

    def walk_fwd_validation_univ(self, model: str, model_cfg, data, n_test):
        """
        :param model: Statistical model
                'ar': Autoregressive model
                'arma': Autoregressive Moving Average model
                'arima': Autoregressive Integrated Moving Average model
                'varmax': Vector Autoregressive Moving Average with eXogenous regressors model
                'hw': Holt Winters model
        :param data:
        :param n_test: number of test data
        :param model_cfg: configuration
        :return:
        """
        # split dataset
        train, test = self.train_test_split(data=data, n_test=n_test)
        # history = data.values  # seed history with training dataset

        predictions = []
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.model_univ[model](history=train,
                                          config=model_cfg,
                                          pred_step=n_test-i)
            # store forecast in list of predictions
            predictions.append(yhat)

            # add actual observation to history for the next loop
            train = np.append(train, test[i])

        # estimate prediction error
        error = self.calc_sqrt_mse(test, predictions)

        return error

    # def grid_search(self, model: str, data, n_test: int, cfg_list: list):
    #     scores = [self.score_model(model=model, data=data, n_test=n_test,
    #                                cfg=cfg) for cfg in cfg_list]
    #     # remove empty results
    #     scores = [score for score in scores if score[1] != None]
    #     # sort configs by error, asc
    #     scores.sort(key=lambda tup: tup[2])
    #
    #     return scores


class ModelLstm(object):
    def __init__(self, sell_in, sell_out):
        self.df_sell_in = sell_in
        self.df_sell_out = sell_out
        # Hyperparameters
        self.lstm_unit = config.LSTM_UNIT
        self.epochs = config.EPOCHS
        self.batch_size = config.BATCH_SIZE

    def lstml_vanilla(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                      units: int, timesteps: int, pred_steps=1):

        x_train, y_train = self.preprocessing.split_sequence_univ(df=train, feature='Close',
                                                                  timesteps=timesteps, pred_steps=pred_steps)
        x_val, y_val = self.preprocessing.split_sequence_univ(df=val, feature='Close',
                                                              timesteps=timesteps, pred_steps=pred_steps)
        x_test, y_test = self.preprocessing.split_sequence_univ(df=test, feature='Close',
                                                                timesteps=timesteps, pred_steps=pred_steps)
        # Reshape
        n_features = 1
        x_train = self.lstm_data_reshape(data=x_train, n_feature=n_features)
        x_val = self.lstm_data_reshape(data=x_val, n_feature=n_features)
        x_test = self.lstm_data_reshape(data=x_test, n_feature=n_features)

        # Build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(timesteps, n_features)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mae')

        history = model.fit(x_train, y_train,
                            epochs=self.__class__.EPOCHS,
                            batch_size=self.__class__.BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            shuffle=False)
        # self.history = history

        predictions = model.predict(x_test, verbose=0)
        rmse = self.calc_sqrt_mse(actual=y_test, predicted=predictions)
        print('Test RMSE: %.3f' % rmse)

        return predictions, rmse

    def lstml_stacked(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                      units: int, timesteps: int, pred_steps=1):

        x_train, y_train = self.preprocessing.split_sequence_univ(df=train, feature='Close',
                                                                  timesteps=timesteps, pred_steps=pred_steps)
        x_val, y_val = self.preprocessing.split_sequence_univ(df=val, feature='Close',
                                                              timesteps=timesteps, pred_steps=pred_steps)
        x_test, y_test = self.preprocessing.split_sequence_univ(df=test, feature='Close',
                                                                timesteps=timesteps, pred_steps=pred_steps)
        # reshape data
        n_features = 1
        x_train = self.lstm_data_reshape(data=x_train, n_feature=n_features)
        x_val = self.lstm_data_reshape(data=x_val, n_feature=n_features)
        x_test = self.lstm_data_reshape(data=x_test, n_feature=n_features)

        # build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', return_sequences=True, input_shape=(timesteps, n_features)))
        model.add(LSTM(units=units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mae')

        # fit model
        history = model.fit(x_train, y_train,
                            epochs=self.__class__.EPOCHS,
                            batch_size=self.__class__.BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            shuffle=False)
        self.history = history

        predictions = model.predict(x_test, verbose=0)
        rmse = self.calc_sqrt_mse(actual=y_test, predicted=predictions)
        print('Test RMSE: %.3f' % rmse)

        return predictions, rmse

    @staticmethod
    def lstm_data_reshape(data: np.array, n_feature: int):
        return data.reshape((data.shape[0], data.shape[1], n_feature))

    @staticmethod
    def calc_sqrt_mse(actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

