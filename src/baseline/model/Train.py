from common.SqlConfig import SqlConfig
from common.SqlSession import SqlSession
from baseline.model.Algorithm import Algorithm
import common.config as config

import warnings

import numpy as np
import pandas as pd
from math import sqrt
from datetime import timedelta
from datetime import datetime

from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


class Train(object):
    def __init__(self, division: str):
        self.sql_config = SqlConfig()
        self.session = SqlSession()
        self.session.init()

        self.end_date = self.session.select(sql=SqlConfig.sql_comm_master(option='RST_END_DAY')).values[0][0]

        self.division = division
        # Hierarchy
        self.hrchy_list = config.HRCHY_LIST
        self.hrchy = config.HRCHY
        self.hrchy_level = config.HRCHY_LEVEL

        # temp information
        self.model_type = 'univ'
        self.time_type = 'W'
        self.target_feature = 'qty'

        self.n_test = config.N_TEST

        # Algorithms
        self.algorithm = Algorithm()
        self.apply_models = ['ar', 'arma', 'arima', 'hw']
        self.model_fn = {'ar': self.algorithm.ar,
                         'arma': self.algorithm.arma,
                         'arima': self.algorithm.arima,
                         'hw': self.algorithm.hw,
                         'var': self.algorithm.var,
                         'varmax': self.algorithm.varmax}

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

    def train(self, df=None, val=None, lvl=0) -> dict:
        temp = None
        if lvl == 0:
            temp = {}
            for key, val in df.items():
                result = self.train(val=val, lvl=lvl+1)
                temp[key] = result

        elif lvl < self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                result = self.train(val=val_hrchy, lvl=lvl+1)
                temp[key_hrchy] = result

            return temp

        elif lvl == self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                if len(val_hrchy):
                    models = self.train_model(df=val_hrchy[self.target_feature])
                temp[key_hrchy] = models

            return temp

        return temp

    def save_score(self, scores: dict):
        result = self.score_to_df(df=scores)

        result = pd.DataFrame(result)
        cols = ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'rmse']
        result.columns = cols

        result['project_cd'] = 'ENT001'
        result['division'] = self.division
        result['fkey'] = ['HRCHY' + str(i+1) for i in range(len(result))]

        result['rmse'] = result['rmse'].fillna(0)

        self.session.insert(df=result, tb_name='M4S_I110410')

    def score_to_df(self, df=None, val=None, lvl=0, hrchy=[]):
        if lvl == 0:
            temp = []
            for key, val in df.items():
                hrchy.append(key)
                result = self.score_to_df(val=val, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key)

        elif lvl < self.hrchy_level:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = self.score_to_df(val=val_hrchy, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        elif lvl == self.hrchy_level:
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                temp = []
                for algorithm, score in val_hrchy:
                    temp.append(hrchy + [algorithm, score])
                hrchy.remove(key_hrchy)

            return temp

        return temp



    def save_prediction(self, predictions):
        end_date = datetime.strptime(self.end_date, '%Y%m%d')

        results = []
        fkey = ['HRCHY' + str(i+1) for i in range(len(predictions))]
        for i, pred in enumerate(predictions):
            for j,  result in enumerate(pred[-1]):
               results.append([fkey[i]] + pred[:-1] + [datetime.strftime(end_date + timedelta(weeks=(j+1)), '%Y%m%d'),
                                           result])

        results = pd.DataFrame(results)
        cols = ['fkey'] + ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'month', 'result_sales']
        results.columns = cols
        results['project_cd'] = 'ENT001'
        results['division'] = self.division

        self.session.insert(df=results, tb_name='M4S_I110400')

    def train_model(self, df) -> tuple:
        models = []
        for model in config.MODEL_CANDIDATES[self.model_type]:
            score = 0
            if self.model_type == 'univ':
                if isinstance(df, pd.DataFrame):    # pandas Dataframe
                    data = df.values
                else:    # numpy array
                    data = df.tolist()
                score = self.walk_fwd_validation_univ(model=model, model_cfg=self.cfg_dict[model],
                                                      data=data, n_test=self.n_test)
            # elif self.model_type == 'multi':
            #     score = self.walk_fwd_validation_multi(model=model, model_cfg=self.cfg_dict.get(model, None),
            #                                            data=df, n_test=self.n_test)
            # elif self.model_type == 'exg':
            #     score = self.lstm_train(train=df, units=config.LSTM_UNIT)

            models.append([model, round(score, 2)])

        models = sorted(models, key=lambda x: x[1])

        return models

    def walk_fwd_validation_univ(self, model: str, model_cfg, data, n_test) -> np.array:
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
            yhat = self.model_fn[model](history=train,
                                        cfg=model_cfg,
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
        result = self.walk_fwd_validation_univ(model=model, data=data, n_test=n_test,
                                               model_cfg=cfg, time_type=self.time_type)
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