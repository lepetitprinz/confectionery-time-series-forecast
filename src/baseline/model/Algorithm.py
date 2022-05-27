import ast
import warnings
import numpy as np

# Uni-variate Statistical Models
from statsmodels.tsa.ar_model import AutoReg    # Auto Regression
from statsmodels.tsa.arima.model import ARIMA    # Auto Regressive Integrated Moving Average
from statsmodels.tsa.holtwinters import SimpleExpSmoothing    # Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing    # Holt-winters Exponential Smoothing

# Multivariate Statistical Models
from statsmodels.tsa.vector_ar.var_model import VAR    # Vector Auto regression
# Vector Autoregressive Moving Average with exogenous regressors model
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX    # Seasonal Auto regressive integrated moving average
# from prophet import Prophet

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


warnings.filterwarnings("ignore")


class Algorithm(object):
    """
    Statistical Model Class

    # Model List
    1. Uni-variate Model
        - AR model (Autoregressive Model)
        - ARIMA model (Autoregressive Integrated Moving Average Model)
        - SES model (Simple Exponential Smoothing Model)
        - HW model (Holt-Winters Exponential Smoothing Model)
        - PROPHET model (Facebook Prophet Model)

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
        :param history: time series history (Pandas Series)
            index: weekly dates
            data: Sales quantity
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
            model_fit = model.fit(cov_type='HC0')

            # Make multi-step prediction
            yhat = model_fit.forecast(steps=pred_step)

        except ValueError:
            yhat = None

        return yhat

    # Autoregressive integrated moving average model
    @staticmethod
    def arima(history, cfg: dict, pred_step=1):
        """
        :param history: time series history (Pandas Series)
            index: weekly dates
            data: Sales quantity
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

        try:
            # fit model
            model_fit = model.fit()

            # Make multi-step forecast
            yhat = model_fit.forecast(steps=pred_step)

        except ValueError:
            yhat = None

        return yhat

    @staticmethod
    def ses(history, cfg: dict, pred_step=1):
        """
        :param history: time series history (Pandas Series)
            index: weekly dates
            data: Sales quantity
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

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step - 1)

        return yhat

    @staticmethod
    def hw(history, cfg: dict, pred_step=1):
        """
        :param history: time series history (Pandas Series)
            index: weekly dates
            data: Sales quantity
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
            smoothing_level : float
            smoothing_trend : float
            smoothing_seasonal : float
        :param pred_step: prediction steps
        :return: forecast result
        """
        # define model
        model = ExponentialSmoothing(history, trend=cfg['trend'],
                                     damped_trend=bool(cfg['damped_trend']),
                                     seasonal=cfg['seasonal'],
                                     seasonal_periods=ast.literal_eval(cfg['seasonal_period']))

        # fit model
        model_fit = model.fit(
            smoothing_level=float(cfg['alpha']),
            smoothing_trend=float(cfg['beta']),
            smoothing_seasonal=float(cfg['gamma']),
            optimized=True,
            remove_bias=bool(cfg['remove_bias'])
        )

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step)

        # Convert nan values to zeros
        yhat = yhat.fillna(0)

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
        endog = np.hstack([history['endog'].reshape(-1, 1), history['exog']])

        # define model
        model = VARMAX(
            endog=endog,
            order=order,
            trend=cfg['trend']
        )

        try:
            # fit model
            model_fit = model.fit()

            # Make multi-step forecast
            yhat = model_fit.forecast(steps=pred_step)
            yhat = yhat[:, 0]

        except ValueError:
            yhat = None

        return yhat

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
        model_fit = model.fit(disp=False)

        # Make multi-step forecast
        yhat = model_fit.forecast(steps=pred_step, exog=[history['exog'][-1]] * pred_step)

        return yhat

    # # Facebook Prophet
    # @staticmethod
    # def prophet(history: dict, cfg: dict, pred_step=1):
    #     """
    #     :param history:
    #         endog: The observed time-series process
    #     :param cfg:
    #         changepoint_prior_scale: Adjusting trend flexibility (0 ~ 1)
    #         seasonality_mode: (multiplicative)
    #         weekly_seasonality: Weekly seasonality (True/False)
    #         interval_width: width of the uncertainty interval (0 ~ 1)
    #         mcmc_samples: uncertainty in seasonality
    #     """
    #     train = pd.DataFrame({'ds': history.index, 'y': history.values})
    #     _ = cfg
    #
    #     with suppress_stdout_stderr():
    #         model = Prophet(
    #             weekly_seasonality=True,
    #             daily_seasonality=False,
    #         )
    #
    #         # model = Prophet(
    #         #     changepoint_prior_scale=cfg['changepoint_prior_scale'],
    #         #     seasonality_mode=cfg['seasonality_mode'],
    #         #     weekly_seasonality=cfg['weekly_seasonality'],
    #         #     interval_width=cfg['interval_width'],
    #         #     mcmc_samples=cfg['mcmc_samples']
    #         # )
    #
    #         # fit model
    #         try:
    #             model.fit(train)
    #
    #             # forecast
    #             future = model.make_future_dataframe(periods=pred_step)
    #             forecast = model.predict(future)
    #             yhat = forecast['yhat'][-pred_step:]
    #
    #         except ValueError:
    #             yhat = None
    #
    #     return yhat

    # Random Forest
    @staticmethod
    def random_forest(data: dict, cfg: dict):
        regr = RandomForestRegressor(
            n_estimators=ast.literal_eval(cfg['n_estimators']),
            criterion=cfg['criterion'],
            max_features=cfg['max_features']
        )
        regr.fit(data['x_train'], data['y_train'])

        yhat = regr.predict(data['x_test'])

        return yhat

    # Random Forest
    @staticmethod
    def gradient_boost(data: dict, cfg: dict):
        regr = GradientBoostingRegressor(
            n_estimators=ast.literal_eval(cfg['n_estimators']),
            criterion=cfg['criterion'],
            max_features=cfg['max_features']
        )
        regr.fit(data['x_train'], data['y_train'])

        yhat = regr.predict(data['x_test'])

        return yhat

    # Extremely Randomized Trees
    @staticmethod
    def extra_trees(data: dict, cfg: dict):
        regr = ExtraTreesRegressor(
            n_estimators=ast.literal_eval(cfg['n_estimators']),
            criterion=cfg['criterion'],
            max_features=cfg['max_features'],
            min_samples_split=cfg['min_samples_split'],
            min_samples_leaf=cfg['min_samples_leaf']
        )
        regr.fit(data['x_train'], data['y_train'])

        yhat = regr.predict(data['x_test'])

        return yhat

    @staticmethod
    def weight_moving_avg(data, weight):
        pass