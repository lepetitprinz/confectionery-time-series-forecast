from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import pickle
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Simulate(object):
    week_standard = 'end'    # start / end
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
        pred, sales = self.preprocessing(discount=discount, hrchy_code=hrchy_code)

        pred = self.lagging(data=pred)

        data = self.add_prev_sales(pred=pred, sales=sales)

        result = self.predict(data=data, hrchy_code=hrchy_code)

        return result

    def add_prev_sales(self, pred: pd.DataFrame, sales: pd.DataFrame):
        pred.loc[sales[self.date_col], 'qty_lag'] = sales['qty_lag'].values

        return pred

    def preprocessing(self, discount: pd.DataFrame, hrchy_code: str):
        # load prediction data
        pred = self.load_prediction(hrchy_code=hrchy_code)

        # merge discount data
        pred = pd.merge(pred, discount, on='yymmdd')

        # convert to datetime type
        pred[self.date_col] = pd.to_datetime(pred[self.date_col], format='%Y%m%d')

        # get first day
        date_sorted = pred[self.date_col].sort_values()
        first_day = date_sorted[0]

        # get days of previous week
        date_dict = self.get_prev_week_days(date=first_day)

        # load sales data
        sales = self.load_sales(date=date_dict, hrchy_code=hrchy_code)

        sales[self.date_col] = pd.to_datetime(sales[self.date_col], format='%Y%m%d')
        sales = sales.set_index(keys=self.date_col)

        sales = sales.resample(rule='W').sum()
        sales = sales.reset_index()
        sales[self.date_col] = sales[self.date_col] + timedelta(days=7)

        pred = pred.set_index(keys=self.date_col)

        return pred, sales

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
        path = os.path.join('..', '..', 'simulation', 'best_models', self.data_version + '_' + self.division_cd +
                            '_' + '4' + '_' + hrchy_code + '.pickle')
        f = open(path, 'rb')
        estimator = pickle.load(f)

        return estimator

    def load_prediction(self, hrchy_code: str):
        pred_conf = {
            'data_vrsn_cd': self.data_version,
            'division_cd': self.division_cd.upper(),
            'fkey': self.fkey_map[self.hrchy_lvl],
            'item_cd': hrchy_code
        }
        prediction = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_item(**pred_conf))

        return prediction

    def load_sales(self, date: dict, hrchy_code: str):
        sales_conf = {
            'item_cd': hrchy_code,
            'from_date': str(date['from_date']),
            'to_date': str(date['to_date'])
        }
        sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_item(**sales_conf))

        return sales

    def lagging(self, data):
        lagged = deepcopy(data[self.target_col])
        lagged = lagged.shift(periods=self.lag_option[self.lag])
        lagged = pd.DataFrame(lagged.values, columns=[self.target_col + '_lag'], index=lagged.index)
        result = pd.concat([data[self.input_col], lagged], axis=1)
        # result = result.fillna(0)

        return result

    def get_prev_week_days(self, date: datetime.date) -> dict:
        date_before_week = date - timedelta(days=7)

        from_date, to_date = ('', '')
        if self.week_standard == 'start':
            from_date = date_before_week
            to_date = date_before_week + timedelta(days=6)
        elif self.week_standard == 'end':
            from_date = date_before_week - timedelta(days=6)
            to_date = date_before_week

        from_date = datetime.strftime(from_date, '%Y%m%d')
        to_date = datetime.strftime(to_date, '%Y%m%d')
        date_dict = {'from_date': from_date, 'to_date': to_date}

        return date_dict
