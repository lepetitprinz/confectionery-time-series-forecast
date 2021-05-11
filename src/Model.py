from DataPrep import DataPrep
import config

from typing import Dict, Callable, Any
import os
import warnings

from copy import deepcopy
import numpy as np
import pandas as pd
from math import sqrt
from datetime import timedelta

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Univariate Statistical Models
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# from prophet import Prophet
warnings.filterwarnings('ignore')


# Model Decorator
def stats_method(func: Callable) -> Callable[[object, list, tuple, int, str], Any]:
    def wrapper(obj: object, history: list, cfg: tuple, time_type: str, pred_step: int):
        return func(obj, history, cfg, time_type, pred_step)
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
        # path
        self.save_path = os.path.join('..', 'result', 'forecast')
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
        self.model_multi: Dict[str, stats_method] = {'var': self.var}
        self.model_exg: Dict[str, stats_method] = {'lstm_vn': self.lstml_train}

        # hyper-parameters
        self.cfg_ar = (config.LAG, config.TREND, config.SEASONAL, config.PERIOD)
        self.cfg_arma = (config.TWO_LVL_ORDER, config.FREQUENCY, config.TREND_ARMA)
        self.cfg_arima = (config.THR_LVL_ORDER, config.FREQUENCY, config.TREND_ARMA)
        self.cfg_ses = (config.INIT_METHOD, config.SMOOTHING, config.OPTIMIZED)
        self.cfg_hw = (config.TREND_HW, config.DAMPED_TREND, config.SEASONAL_HW, config.PERIOD,
                       config.USE_BOXCOX, config.REMOVE_BIAS)
        self.cfg_var = config.LAG
        self.cfg_varma = (config.TWO_LVL_ORDER, config.TREND)
        self.cfg_varmax = (config.TWO_LVL_ORDER, config.TREND)

        self.cfg_dict = {'ar': self.cfg_ar,
                         'arma': self.cfg_arma,
                         'arima': self.cfg_arima,
                         'ses': self.cfg_ses,
                         'hw': self.cfg_hw,
                         'var': self.cfg_var,
                         'varma': self.cfg_varma,
                         'varmax': self.cfg_varmax}

        self.epoch_best = 0

    def train(self) -> dict:
        print("Implement model training")
        scores = pd.DataFrame()
        best_models = deepcopy(self.df_sell)

        for sell_type, cust in best_models.items():   # sell type: sell-in / sell-out
            for cust_type, prod in cust.items():    # customer type: all / A / B / C
                for prod_type, time in prod.items():    # product type: all / 가 / 나 / 다 / 라 / 바
                    for time_type, df in time.items():    # time type: D / W / M
                        if len(df) > 0:
                            best_model, models = self.get_best_models(df=df, time_type=time_type)
                            time[time_type] = best_model
                            # Make accuracy score dataframe
                            temp = pd.DataFrame({'sell_type': sell_type,
                                                 'cust_type': cust_type,
                                                 'prod_type': prod_type,
                                                 'time_type': time_type,
                                                 'model': np.array(models)[:, 0],
                                                 'rmse': np.array(models)[:, 1]})
                            scores = pd.concat([scores, temp])

        scores.to_csv(os.path.join(self.save_path, 'scores_' + self.model_type + '.csv'),
                      index=False, encoding='utf-8-sig')
        print("Training model is finished\n")

        return best_models

    def forecast(self, best_models: dict) -> dict:
        print("Implement model prediction")
        forecast = deepcopy(self.df_sell)
        for sell_type, cust in forecast.items():  # sell type: sell-in / sell-out
            for cust_type, prod in cust.items():  # customer type: all / A / B / C
                for prod_type, time in prod.items():  # product type: all / 가 / 나 / 다 / 라 / 바
                    for time_type, df in time.items():  # time type: D / W
                        best_model = best_models[sell_type][cust_type][prod_type][time_type][0]
                        prediction = None
                        if self.model_type == 'univ':
                            prediction = self.model_univ[best_model](history=df,
                                                                     cfg=self.cfg_dict[best_model],
                                                                     time_type=time_type,
                                                                     pred_step=config.N_TEST)
                        elif self.model_type == 'multi':
                            prediction = self.model_multi[best_model](history=df,
                                                                      cfg=self.cfg_dict[best_model],
                                                                      time_type=time_type,
                                                                      pred_step=config.N_TEST)
                        elif self.model_type == 'exg':
                            prediction = self.lstml_predict(train=df, units=config.LSTM_UNIT)

                        time[time_type] = np.round(prediction)

        print("Model prediction is finished\n")

        return forecast

    def forecast_all(self) -> dict:
        print("Implement model prediction")
        forecast = deepcopy(self.df_sell)
        for sell_type, cust in forecast.items():  # sell type: sell-in / sell-out
            for cust_type, prod in cust.items():  # customer type: all / A / B / C
                for prod_type, time in prod.items():  # product type: all / 가 / 나 / 다 / 라 / 바
                    for time_type, df in time.items():  # time type: D / W
                        pred_all = {}
                        if self.model_type == 'univ':
                            for model in self.model_univ:
                                prediction = self.model_univ[model](history=df,
                                                                    cfg=self.cfg_dict[model],
                                                                    time_type=time_type,
                                                                    pred_step=config.N_TEST)
                                pred_all[model] = np.round(prediction)
                        elif self.model_type == 'multi':
                            for model in self.model_multi:
                                prediction = self.model_multi[model](history=df,
                                                                     cfg=self.cfg_dict[model],
                                                                     time_type=time_type,
                                                                     pred_step=config.N_TEST)
                                pred_all[model] = np.round(prediction)
                        elif self.model_type == 'exg':
                            prediction = self.lstml_predict(train=df, units=config.LSTM_UNIT)
                            pred_all['lstm'] = np.round(prediction)

                        time[time_type] = pred_all

        print("Model prediction is finished\n")

        return forecast

    def save_result(self, forecast: dict, best_models) -> None:
        result = pd.DataFrame()
        for sell_type, cust in forecast.items():
            for cust_type, prod in cust.items():  # customer type: all / A / B / C
                for prod_type, time in prod.items():  # product type: all / 가 / 나 / 다 / 라 / 바
                    for time_type, prediction in time.items():  # time type: D / W
                        start_dt = self.df_sell[sell_type][cust_type][prod_type][time_type].index[-1]
                        start_dt += timedelta(days=1)
                        dt = pd.date_range(start=start_dt, periods=config.N_TEST, freq=time_type)
                        best_model = best_models[sell_type][cust_type][prod_type][time_type][0]
                        temp = pd.DataFrame({'sell_type': sell_type,
                                             'cust_type': cust_type,
                                             'prod_type': prod_type,
                                             'time_type': time_type,
                                             'model': best_model,
                                             'dt': dt,
                                             'prediction': prediction})
                        result = pd.concat([result, temp])

        result.to_csv(os.path.join(self.save_path, 'forecast_' + self.model_type + '.csv'),
                      index=False, encoding='utf-8-sig')
        print("Forecast results are saved\n")

    def save_result_all(self, forecast: dict) -> None:
        result = pd.DataFrame()
        for sell_type, cust in forecast.items():
            for cust_type, prod in cust.items():  # customer type: all / A / B / C
                for prod_type, time in prod.items():  # product type: all / 가 / 나 / 다 / 라 / 바
                    for time_type, prediction in time.items():  # time type: D / W
                        for model, pred_result in prediction.items():
                            start_dt = self.df_sell[sell_type][cust_type][prod_type][time_type].index[-1]
                            start_dt += timedelta(days=1)
                            dt = pd.date_range(start=start_dt, periods=config.N_TEST, freq=time_type)
                            temp = pd.DataFrame({'sell_type': sell_type,
                                                 'cust_type': cust_type,
                                                 'prod_type': prod_type,
                                                 'time_type': time_type,
                                                 'model': model,
                                                 'dt': dt,
                                                 'prediction': pred_result})
                            result = pd.concat([result, temp])

        result.to_csv(os.path.join(self.save_path, 'forecast_all_' + self.model_type + '.csv'),
                      index=False, encoding='utf-8-sig')

        print("Forecast results are saved\n")

    def get_best_models(self, df, time_type: str) -> tuple:
        best_models = []
        for model in config.MODEL_CANDIDATES[self.model_type]:
            score = 0
            if self.model_type == 'univ':
                if isinstance(df, pd.DataFrame):
                    data = df.values
                else:
                    data = df.tolist()
                score = self.walk_fwd_validation_univ(model=model, model_cfg=self.cfg_dict[model],
                                                      data=data, n_test=config.N_TEST, time_type=time_type)
            elif self.model_type == 'multi':
                score = self.walk_fwd_validation_multi(model=model, model_cfg=self.cfg_dict.get(model, None),
                                                       data=df, n_test=config.N_TEST, time_type=time_type)

            elif self.model_type == 'exg':
                # train_size = int(len(df) * config.TRAIN_RATE)
                # train = df.iloc[:train_size, :]
                # val = df.iloc[train_size:, :]

                score = self.lstml_train(train=df, units=config.LSTM_UNIT)

            if self.model_type != 'exg':
                best_models.append([model, round(score)])
            else:
                best_models.append([model, score])

        # prophet model
        # score = self.prophet(history=df, n_test=config.N_TEST)
        # best_models.append(['prophet', score])

        best_models = sorted(best_models, key=lambda x: x[1])

        return best_models[0], best_models

    def walk_fwd_validation_univ(self, model: str, model_cfg, data, n_test, time_type) -> np.array:
        """
        :param model: Statistical model
        :param model_cfg: configuration
        :param data: time series data
        :param n_test: number of test data
        :param time_type: Data Time range
        :return:
        """
        # split dataset
        train, test = self.train_test_split(data=data, train_size=config.TRAIN_RATE)
        # history = data.values  # seed history with training dataset

        predictions = []
        for i in range(len(test) - n_test + 1):
            # fit model and make forecast for history
            yhat = self.model_univ[model](history=train,
                                          cfg=model_cfg,
                                          pred_step=n_test,
                                          time_type=time_type)
            # store err in list of predictions
            err = self.calc_sqrt_mse(test[i: i+n_test], yhat)
            predictions.append(err)

            # add actual observation to history for the next loop
            train = np.append(train, test[i])

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def walk_fwd_validation_multi(self, model: str, model_cfg, data, n_test, time_type) -> np.array:
        # split dataset
        train, test = self.train_test_split(data=data, train_size=config.TRAIN_RATE)
        # history = data.values  # seed history with training dataset

        predictions = []
        for i in range(len(test) - n_test + 1):
            # fit model and make forecast for history
            yhat = self.model_multi[model](history=train,
                                           cfg=model_cfg,
                                           time_type=time_type,
                                           pred_step=n_test)
            # store err in list of predictions
            err = self.calc_sqrt_mse(test[config.TARGET_COL][i: i+n_test], yhat)
            predictions.append(err)

            # add actual observation to history for the next loop
            train = train.append(test.iloc[i, :])

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def train_test_split(self, data, train_size):
        data_length = len(data)
        if self.model_type == 'univ':
            return data[: int(data_length * train_size)], data[int(data_length * train_size):]

        elif self.model_type == 'multi':
            return data.iloc[:int(data_length * train_size), :], data.iloc[int(data_length * train_size):, :]

    @staticmethod
    def calc_sqrt_mse(actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    def score_model(self, model: str, data, n_test, cfg):
        # convert config to a key
        key = str(cfg)
        result = self.walk_fwd_validation_univ(model=model, data=data, n_test=n_test,
                                               model_cfg=cfg, time_type=config.RESAMPLE_RULE)
        return model, key, result

    # def grid_search(self, model: str, data, n_test: int, cfg_list: list):
    #     scores = [self.score_model(model=model, data=data, n_test=n_test,
    #                                cfg=cfg) for cfg in cfg_list]
    #     # remove empty results
    #     scores = [score for score in scores if score[1] != None]
    #     # sort configs by error, asc
    #     scores.sort(key=lambda tup: tup[2])
    #
    #     return scores

    #############################
    # Univariate Model
    #############################
    # Auto-regressive Model
    @stats_method
    def ar(self, history, cfg: tuple, time_type, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                l: lags (int)
                t: trend ('c', 'nc')
                    c: Constant
                    nc: No Constant
                s: seasonal (bool)
                p: period (Only used if seasonal is True)
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step: prediction steps
        :return: forecast result
        """
        lag, t, s, p = cfg
        model = AutoReg(endog=history, lags=lag[time_type], trend=t, seasonal=s, period=p[time_type])
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step prediction
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    # Auto regressive moving average model
    @stats_method
    def arma(self, history, cfg: tuple, time_type, pred_step=0):
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
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = cfg
        # define model
        model = ARMA(endog=history, order=o[time_type], freq=f)
        # fit model
        model_fit = model.fit(trend=t, disp=0)
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    # Autoregressive integrated moving average model
    @stats_method
    def arima(self, history, cfg: tuple, time_type, pred_step=0):
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
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = cfg
        # define model
        model = ARIMA(history, order=o[time_type], trend=t, freq=f)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @stats_method
    def ses(self, history, cfg: tuple, time_type, pred_step=0):
        """
        :param history: time series data
        :param cfg:
                i: initialization_method (None, 'estimated', 'heuristic', 'legacy-heuristic', 'known')
                    - Method for initialize the recursions
                l: smoothing level (float)
                o: optimized or not (bool)
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step: prediction steps
        :return: forecast result
        """
        i, s, o = cfg
        _ = time_type
        # define model
        model = SimpleExpSmoothing(history, initialization_method=i)
        # fit model
        model_fit = model.fit(smoothing_level=s, optimized=o)     # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @stats_method
    def hw(self, history, cfg: tuple, time_type, pred_step=0):
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
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step: prediction steps
        :return: forecast result
        """
        t, d, s, p, b, r = cfg
        # define model
        model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s,
                                     seasonal_periods=p[time_type], use_boxcox=b)
        # fit model
        model_fit = model.fit(optimized=True, remove_bias=r)     # fit model
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
    @stats_method
    def var(self, history: pd.DataFrame, cfg, time_type, pred_step=0):
        """
        :param history:
        :param cfg:
        :param time_type: Data Time range
                M: month
                W: week
                D: day
        :param pred_step:
        :return:
        """
        lag = cfg
        # define model
        model = VAR(history)
        # fit model
        model_fit = model.fit(lag[time_type])
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        # yhat = model.predict(start=len(history), end=len(history) + pred_step - 1)
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    # Vector Autoregressive Moving Average model
    @staticmethod
    def varma(history: pd.DataFrame, cfg, pred_step=0):
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
        model = VARMAX(history.astype(float),  order=o, trend=t)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    # Vector Autoregressive Moving Average with eXogenous regressors model
    @staticmethod
    def varmax(history: pd.DataFrame, data_exog: pd.DataFrame, cfg, pred_step=0):
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
        model = VARMAX(history, exog=data_exog,  order=o, trend=t)
        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(y=history.values, steps=pred_step)

        return yhat[:, 0]

    def lstml_train(self, train: pd.DataFrame, units: int):
        # scaling
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)
        train_scaled = pd.DataFrame(train_scaled, columns=train.columns)

        x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
                                                   n_steps_out=config.N_TEST)

        n_features = x_train.shape[2]

        # Build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', return_sequences=True,
                  input_shape=(config.N_TEST, n_features)))
        # model.add(LSTM(units=units, activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss=self.root_mean_squared_error)

        history = model.fit(x_train, y_train,
                            epochs=config.EPOCHS,
                            batch_size=config.BATCH_SIZE,
                            validation_split=1-config.TRAIN_RATE,
                            shuffle=False,
                            verbose=0)
        self.epoch_best = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

        rmse = min(history.history['val_loss'])

        return rmse

    def lstml_predict(self, train: pd.DataFrame, units: int):
        # scaling
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)
        train_scaled = pd.DataFrame(train_scaled, columns=train.columns)

        x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
                                                   n_steps_out=config.N_TEST)
        n_features = x_train.shape[2]

        # Build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', return_sequences=True,
                  input_shape=(config.N_TEST, n_features)))
        # model.add(LSTM(units=units, activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss=self.root_mean_squared_error)

        model.fit(x_train, y_train,
                  epochs=self.epoch_best,
                  batch_size=config.BATCH_SIZE,
                  shuffle=False,
                  verbose=0)
        test = x_train[-1]
        test = test.reshape(1, test.shape[0], test.shape[1])

        predictions = model.predict(test, verbose=0)
        predictions = scaler.inverse_transform(predictions[0])

        return predictions[:, 0]

    @staticmethod
    def lstm_data_reshape(data: np.array, n_feature: int):
        return data.reshape((data.shape[0], data.shape[1], n_feature))

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
