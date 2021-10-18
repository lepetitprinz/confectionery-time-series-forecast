from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import pickle
import pandas as pd
from copy import deepcopy

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Simulate(object):
    fkey_map = {3: 'C0-P3', 4: 'C0-P4', 5: 'C0-P5', 6: 'C1-P5'}
    lag_option = {'w1': 1, 'w2': 2}
    estimators = {'rf': RandomForestRegressor,
                  'gb': GradientBoostingRegressor,
                  'et': ExtraTreesRegressor}

    def __init__(self, data_version: str, division_cd: str, hrchy_lvl: int, lag: str,
                 scaling_yn: bool, save_obj_yn: bool):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Data Configuration
        self.data_version = data_version
        self.division_cd = division_cd
        # self.hrchy_lvl = hrchy_lvl
        self.hrchy_lvl = 5
        self.date_col = 'yymmdd'
        self.input_col = 'discount'
        self.target_col = 'qty'
        self.lag = lag

        # Prediction Option configuration
        self.scaling_yn = scaling_yn
        self.save_obj_yn = save_obj_yn

    def simulate(self, discount, hrchy_code):
        prediction = self.load_prediction(hrchy_code=hrchy_code)
        data = pd.merge(prediction, discount, on='yymmdd')
        data[self.date_col] = pd.to_datetime(data[self.date_col], format='%Y%m%d')
        data = data.set_index(keys=self.date_col)

        data = self.lagging(data=data)

        result = self.predict(data=data, hrchy_code=hrchy_code)

        return result

    def predict(self, data, hrchy_code):
        # Load best estimator
        estimator_info = self.load_best_estimator(hrchy_code=hrchy_code)
        estimator = estimator_info[3]

        # Predict
        result = estimator.predict(data)

        return result

    def load_best_estimator(self, hrchy_code: str):
        # path = os.path.join(self.data_version + '_' + self.division_cd + '_' + str(self.hrchy_lvl) +
        #                     '_' + hrchy_code + '.pickle')
        path = os.path.join('..', '..', 'simulation', 'best_models',
            self.data_version + '_' + self.division_cd + '_' + '4' + '_' + hrchy_code + '.pickle')
        f = open(path, 'rb')
        estimator = pickle.load(f)

        return estimator

    def load_sales(self, week, hrchy_code: str):
        sales_conf = {
            'item_cd': hrchy_code,
            'week': week.upper()
        }
        sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_item(**sales_conf))

        return sales

    def load_prediction(self, hrchy_code: str):
        pred_conf = {
            'data_vrsn_cd': self.data_version,
            'division_cd': self.division_cd.upper(),
            'fkey': self.fkey_map[self.hrchy_lvl],
            'item_cd': hrchy_code
        }
        prediction = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_item(**pred_conf))

        return prediction

    def lagging(self, data):
        lagged = deepcopy(data[self.target_col])
        lagged = lagged.shift(periods=self.lag_option[self.lag])
        lagged = pd.DataFrame(lagged.values, columns=[self.target_col + '_lag'], index=lagged.index)
        result = pd.concat([data[self.input_col], lagged], axis=1)
        result = result.fillna(0)

        return result