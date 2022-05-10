from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Simulate(object):
    week_standard = 'start'    # start / end
    fkey_map = {3: 'C0-P3', 4: 'C0-P4', 5: 'C0-P5', 6: 'C1-P5'}
    lag_option = {'w1': 1, 'w2': 2}
    estimators = {'rf': RandomForestRegressor,
                  'gb': GradientBoostingRegressor,
                  'et': ExtraTreesRegressor}

    # Rename columns
    hrchy_cd_to_db_cd_map = {
        'biz_cd': 'item_attr01_cd', 'line_cd': 'item_attr02_cd', 'brand_cd': 'item_attr03_cd',
        'item_cd': 'item_attr04_cd', 'biz_nm': 'item_attr01_nm', 'line_nm': 'item_attr02_nm',
        'brand_nm': 'item_attr03_nm', 'item_nm': 'item_attr04_nm'
    }
    hrchy_sku_to_db_sku_map = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}

    def __init__(
            self,
            data_version: str,
            division_cd: str,
            cust_grp_cd: str,
            date: dict,
            lag: str,
            exec_cfg: dict,
            item_cd,
            discount
    ):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Execute Configuration
        self.exec_cfg = exec_cfg
        self.path_root = os.path.join('..', '..', 'simulation', 'model')

        # Data Configuration
        self.data_version = data_version
        self.division_cd = division_cd
        self.cust_grp_cd = cust_grp_cd
        self.date = date
        self.date_range = []
        self.cal = None
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())
        self.input_col = 'discount'
        self.date_col = self.common['date_col']
        self.target_col = self.common['target_col']
        self.lag = lag
        self.hrchy_lvl = 6
        self.item_cd = item_cd
        self.discount = discount
        self.tb_name = 'M4S_I110500'

    def simulate(self):
        self.set_date_range()

        pred, sales = self.preprocessing(discount=self.discount, hrchy_code=self.item_cd)

        pred = self.lagging(data=pred)

        data = self.add_prev_sales(pred=pred, sales=sales)

        result = self.predict(data=data, hrchy_code=self.item_cd)

        return result

    def set_date_range(self):
        date_list = list(pd.date_range(start=self.date['date_from'], end=self.date['date_to'], freq='W'))
        date_list = [datetime.strftime(date - timedelta(days=6), '%Y%m%d') for date in date_list]

        self.date_range = date_list

    def preprocessing(self, discount: pd.DataFrame, hrchy_code: str):
        # load prediction data
        pred = self.load_prediction(hrchy_code=hrchy_code)

        # slice the data set
        pred = pred[pred[self.date_col].isin(self.date_range)]

        # add discount data
        pred['discount'] = discount
        # pred = pd.merge(pred, discount, on=self.date_col)

        self.cal = pred[[self.date_col, 'week']]

        # convert to datetime type
        pred[self.date_col] = pd.to_datetime(pred[self.date_col], format='%Y%m%d')

        # get first day
        date_sorted = pred[self.date_col].sort_values()
        first_day = date_sorted[0]

        # get days of previous week
        date_dict = self.get_prev_week_days(date=first_day)

        # load sales data
        sales = self.load_sales(date=date_dict, hrchy_code=hrchy_code)

        if len(sales) > 0:
            sales = sales['qty_lag'].sum()

        pred = pred.set_index(keys=self.date_col)

        return pred, sales

    @staticmethod
    def add_prev_sales(pred: pd.DataFrame, sales: float):
        pred.loc[:, 'qty_lag'][0] = sales

        return pred

    def predict(self, data, hrchy_code):
        # Load best estimator
        estimator_info = self.load_best_estimator(hrchy_code=hrchy_code)
        estimator = estimator_info[3]

        # Predict
        result = estimator.predict(data)
        result = np.round(result, 2)

        return result

    def load_best_estimator(self, hrchy_code: str):
        path = os.path.join(self.path_root,  self.division_cd + '_' + self.data_version +
                            '_' + self.cust_grp_cd + '-' + hrchy_code + '.pickle')
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
            'division_cd': self.division_cd,
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

    def save_result(self, result: list):
        df, info = self.make_db_format(result=result)
        if self.exec_cfg['save_db_yn']:
            self.io.delete_from_db(sql=self.sql_conf.del_sim_result(**info))
            self.io.insert_to_db(df=df, tb_name=self.tb_name)

        return result

    def make_db_format(self, result):
        project_cd = self.common['project_cd']
        data_vrsn_cd = self.data_version

        df = pd.DataFrame({'result_sales': result})
        df['apply_disc'] = self.discount
        df['project_cd'] = project_cd
        df['data_vrsn_cd'] = data_vrsn_cd
        df['division_cd'] = self.division_cd
        df[self.date_col] = self.cal[self.date_col]
        df['yy'] = df['yymmdd'].str.slice(stop=4)
        df['week'] = self.cal['week']
        df['item_cd'] = self.item_cd

        #
        item_mst = self.item_mst.rename(columns=self.hrchy_cd_to_db_cd_map)
        item_mst = item_mst.rename(columns=self.hrchy_sku_to_db_sku_map)
        df = pd.merge(df, item_mst, how='left', on='item_cd')

        info = {
            'project_cd': project_cd,
            'data_vrsn_cd': data_vrsn_cd,
            'division_cd': self.division_cd,
            'item_cd': self.item_cd
        }

        return df, info
