import common.config as config
import common.util as util
from common.SqlConfig import SqlConfig
from common.SqlSession import SqlSession
from baseline.model.Algorithm import Algorithm

import warnings

import numpy as np
import pandas as pd
from math import sqrt
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


class Train(object):
    def __init__(self, division: str, cand_models: list, param_grid: dict, end_date):
        # Data Configuration
        self.division = division    # SELL-IN / SELL-OUT
        self.end_date = end_date
        self.target = 'qty'

        # Hierarchy
        self.hrchy_list = config.HRCHY_LIST
        self.hrchy = config.HRCHY
        self.hrchy_level = config.HRCHY_LEVEL

        self.n_test = config.N_TEST

        # Algorithms
        self.algorithm = Algorithm()
        self.cand_models = cand_models
        self.param_grid = param_grid
        self.model_to_variate = config.MODEL_TO_VARIATE
        self.model_fn = {'ar': self.algorithm.ar,
                         'arima': self.algorithm.arima,
                         'hw': self.algorithm.hw,
                         'var': self.algorithm.var}

        self.epoch_best = 0

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

    def train(self, df) -> dict:
        scores = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                      fn=self.train_model,
                                      df=df)

        return scores

    def make_score_result(self, scores: dict) -> pd.DataFrame:
        result = util.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_level,
                                               fn=self.score_to_df,
                                               df=scores)

        result = pd.DataFrame(result)
        cols = ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'rmse']
        result.columns = cols

        result['project_cd'] = 'ENT001'
        result['division'] = self.division
        result['fkey'] = ['HRCHY' + str(i+1) for i in range(len(result))]

        result['rmse'] = result['rmse'].fillna(0)

        return result

    @staticmethod
    def score_to_df(hrchy: list, data):
        result = []
        for algorithm, score in data:
            result.append(hrchy + [algorithm, score])

        return result

    def make_pred_result(self, predictions):
        end_date = datetime.strptime(self.end_date, '%Y%m%d')

        results = []
        fkey = ['HRCHY' + str(i+1) for i in range(len(predictions))]
        for i, pred in enumerate(predictions):
            for j,  result in enumerate(pred[-1]):
                results.append([fkey[i]] + pred[:-1] +
                              [datetime.strftime(end_date + timedelta(weeks=(j+1)), '%Y%m%d'), result])

        results = pd.DataFrame(results)
        cols = ['fkey'] + ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'month', 'result_sales']
        results.columns = cols
        results['project_cd'] = 'ENT001'
        results['division'] = self.division

        return results

    def train_model(self, df) -> tuple:
        models = []
        for model in self.cand_models:
            data = self.filter_data(df=df, model=model)
            score = self.walk_fwd_validation(model=model, cfg=self.param_grid[model],
                                             data=data, n_test=self.n_test)

            models.append([model, round(score, 2)])

        models = sorted(models, key=lambda x: x[1])

        return models

    def filter_data(self, df: pd.DataFrame, model: str):
        filtered = None
        if self.model_to_variate[model] == 'univ':
            filtered = df[self.target]
        elif self.model_to_variate[model] == 'multi':
            pass

        return filtered

    def walk_fwd_validation(self, model: str, cfg, data, n_test) -> np.array:
        """
        :param model: Statistical model
        :param cfg: configuration
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
            yhat = self.model_fn[model](history=train,
                                        cfg=cfg,
                                        pred_step=n_test)
            # store err in list of predictions
            err = self.calc_sqrt_mse(test[i: i+n_test], yhat)
            predictions.append(err)

            # add actual observation to history for the next loop
            # train = np.append(train, test[i])

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def walk_fwd_validation_multi(self, model: str, model_cfg, data, n_test) -> np.array:
        # split dataset
        train, test = self.train_test_split(data=data, train_size=config.TRAIN_RATE)
        # history = data.values  # seed history with training dataset

        predictions = []
        for i in range(len(test) - n_test + 1):
            # fit model and make forecast for history
            yhat = self.model_multi[model](history=train,
                                           cfg=model_cfg,
                                           pred_step=n_test)
            # store err in list of predictions
            err = self.calc_sqrt_mse(test[config.COL_TARGET][i: i + n_test], yhat)
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
    def calc_sqrt_mse(actual, predicted) -> float:
        return sqrt(mean_squared_error(actual, predicted))

    def score_model(self, model: str, data, n_test, cfg) -> tuple:
        # convert config to a key
        key = str(cfg)
        result = self.walk_fwd_validation(model=model, data=data, n_test=n_test,
                                          cfg=cfg, time_type=self.time_type)
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