import ast
import numpy as np

# Uni-variate Statistical Models
from statsmodels.tsa.ar_model import AutoReg    # Auto Regression
from statsmodels.tsa.arima.model import ARIMA    # Auto Regressive Integrated Moving Average
from statsmodels.tsa.holtwinters import SimpleExpSmoothing    # Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing    # Holt-winters Exponential Smoothing

from statsmodels.tsa.vector_ar.var_model import VAR    # Vector Auto regression
from statsmodels.tsa.statespace.varmax import VARMAX    # Vector Autoregressive Moving Average with
                                                        # eXogenous regressors model
from statsmodels.tsa.statespace.sarimax import SARIMAX    # Seasonal Auto regressive integrated moving average

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization
# from tensorflow.keras.layers import RepeatVector, TimeDistributed
# from tensorflow.keras.optimizers import Adam

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
        - AR model (Autoregressive Model)
        - ARIMA model (Autoregressive Integrated Moving Average Model)
        - SES model (Simple Exponential Smoothing Model)
        - HWES model (Holt-Winters Exponential Smoothing Model)

    2. Multi-variate Model
        - VAR model (Vector Autoregressive model)
        - VARMAX model (Vector Autoregressive Moving Average with eXogenous regressors model)
        - SARIMA model (Seasonal Autoregressive integrated Moving Average)
    """

    #############################
    # Uni-variate Model
    #############################
    # Auto-regressive Model
    @staticmethod
    def ar(history, cfg: dict, pred_step=1):
        """
        :param history: time series data
        :param cfg:
                l: lags (int)
                t: trend ('n', 'c', 't', 'ct)
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: No Constant and time trend
                s: seasonal (bool)
                p: period (Only used if seasonal is True)
        :param pred_step: prediction steps
        :return: forecast result
        """
        model = AutoReg(endog=history, lags=ast.literal_eval(cfg['lags']), trend=cfg['trend'],
                        seasonal=bool(cfg['seasonal']), period=ast.literal_eval(cfg['period']))

        try:
            model_fit = model.fit()
            # print('Coefficients: {}'.format(model_fit.params))
            # print(model_fit.summary())

            # Make multi-step prediction
            yhat = model_fit.forecast(steps=pred_step)

        except:
            yhat = None

        return yhat

    # Autoregressive integrated moving average model
    @staticmethod
    def arima(history, cfg: dict, pred_step=1):
        """
        :param history: time series data
        :param cfg:
                order: (p, d, q)
                    p: Trend auto-regression order
                    d: Trend difference order
                    q: Trend moving average order
                freq: frequency of the time series (‘B’, ‘D’, ‘W’, ‘M’, ‘A’, ‘Q)
                trend: 'n', 'c', 't', 'ct'
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        # define model
        order = (ast.literal_eval(cfg['p']), ast.literal_eval(cfg['d']), ast.literal_eval(cfg['q']))
        model = ARIMA(history, order=order, trend=cfg['trend'], freq=ast.literal_eval(cfg['freq']))

        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step)

        return yhat

    @staticmethod
    def ses(history, cfg: dict, pred_step=1):
        """
        :param history: time series data
        :param cfg:
                initialization_method: None, 'estimated', 'heuristic', 'legacy-heuristic', 'known'
                    - Method for initialize the recursions
                smoothing_level: smoothing level (float)
                optimized: optimized not (bool)
        :param pred_step: prediction steps
        :return: forecast result
        """
        # define model
        model = SimpleExpSmoothing(history, initialization_method=cfg['initialization_method'])

        # fit model
        model_fit = model.fit(smoothing_level=cfg['smoothing_level'], optimized=cfg['optimized'])  # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @staticmethod
    def hw(history, cfg: dict, pred_step=1):
        """
        :param history: time series data
        :param cfg:
                trend: 'add', 'mul', 'additive', 'multiplicative'
                    - type of trend component
                damped_trend: bool
                    - should the trend component be damped
                seasonal: 'add', 'mul', 'additive', 'multiplicative', None
                    - Type of seasonal component
                seasonal_periods: int
                    - The number of periods in a complete seasonal cycle
                use_boxcox : True, False, ‘log’, float
                    - Should the Box-Cox transform be applied to the data first?
                remove_bias : bool
                    - Remove bias from forecast values and fitted values by enforcing that the average residual is
                      equal to zero
        :param pred_step: prediction steps
        :return: forecast result
        """
        # define model
        model = ExponentialSmoothing(history, trend=cfg['trend'],
                                     damped_trend=bool(cfg['damped_trend']),
                                     seasonal=cfg['seasonal'],
                                     seasonal_periods=ast.literal_eval(cfg['seasonal_period']))

        # fit model
        model_fit = model.fit(optimized=True, remove_bias=bool(cfg['remove_bias']))  # fit model
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step)

        return yhat

    #############################
    # Multi-variate Model
    #############################
    @staticmethod
    def var(history: dict, cfg: dict, pred_step=1):
        """
        :param history:
            endog: 2-d endogenous response variable
            exog: 2-d exogenous variable
        :param cfg:
                maxlags: int, None
                    Maximum number of lags to check for order selection
                ic: 'aic', 'fpe', 'hqic', 'bic', None
                    Information criterion to use for VAR order selection
                        aic: Akaike
                        fpe: Final prediction error
                        hqic: Hannan-Quinn
                        bic: Bayesian a.k.a. Schwarz
                trend: 'c', 'ct', 'ctt', 'nc', 'n'
                    c: add constant
                    ct: constant and trend
                    ctt: constant, linear and quadratic trend
                    nc: co constant, no trend
                    n: no trend
        :param pred_step:
        :return:
        """
        # define model
        data = np.hstack((history['endog'].reshape(-1, 1), history['exog']))
        model = VAR(data)

        # fit model
        model_fit = model.fit(maxlags=ast.literal_eval(cfg['maxlags']),
                              ic=ast.literal_eval(cfg['ic']), trend=cfg['trend'])
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(y=data, steps=pred_step)

        return yhat[:, 0]

    # Vector Autoregressive Moving Average with eXogenous regressors model
    @staticmethod
    def varmax(history: dict, cfg: dict, pred_step=1):
        """
        :param history:
            endog: 2-d endogenous response variable
            exog: 2-d exogenous variable
        :param cfg:
                order: (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                trend: 'n', 'c', 't', 'ct'
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        order = (ast.literal_eval(cfg['p']), ast.literal_eval(cfg['q']))
        data = np.vstack((history['endog'], history['exog'])).T

        # define model
        model = VARMAX(endog=data, order=order, trend=cfg['trend'])

        # fit model
        model_fit = model.fit()
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step)

        return yhat[:, 0]

    @staticmethod
    def sarimax(history: dict, cfg: dict, pred_step=1):
        """
        :param history:
            endog: The observed time-series process
            exog: Array of exogenous regressors, shaped [nobs x k]
        :param cfg:
                order: (p, d, q)
                    p: Trend auto-regression order
                    d: Trend difference order
                    q: Trend moving average order
                seasonal_order: (p, d, q, s)
                    (p, d, q, s) order of the seasonal component of the model for
                    the AR parameters, differences, MA parameters, and periodicity
                trend: 'n', 'c', 't', 'ct'
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        order = (ast.literal_eval(cfg['p']),
                 ast.literal_eval(cfg['d']),
                 ast.literal_eval(cfg['q']))

        seasonal_order = (ast.literal_eval(cfg['ssn_p']),
                          ast.literal_eval(cfg['ssn_d']),
                          ast.literal_eval(cfg['ssn_q']),
                          ast.literal_eval(cfg['ssn_s']))

        # define model
        model = SARIMAX(endog=history['endog'], exog=history['exog'],
                        order=order, seasonal_order=seasonal_order, trend=cfg['trend'])

        # fit model
        model_fit = model.fit()

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step, exog=[history['exog'][-1]] * pred_step)

        return yhat

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
