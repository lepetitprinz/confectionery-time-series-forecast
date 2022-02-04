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


class SimulateDB(object):
    week_standard = 'start'    # start / end
    fkey_map = {4: 'C1-P3', 5: 'C1-P4', 6: 'C1-P5'}
    lag_option = {'w1': 1, 'w2': 2}
    estimators = {
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor,
        'et': ExtraTreesRegressor
    }

    # Rename columns
    hrchy_cd_to_db_cd_map = {
        'biz_cd': 'item_attr01_cd', 'line_cd': 'item_attr02_cd', 'brand_cd': 'item_attr03_cd',
        'item_cd': 'item_attr04_cd', 'biz_nm': 'item_attr01_nm', 'line_nm': 'item_attr02_nm',
        'brand_nm': 'item_attr03_nm', 'item_nm': 'item_attr04_nm'
    }
    hrchy_sku_to_db_sku_map = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}

    def __init__(self, lag: str, exec_cfg: dict, path_root: str):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Execute Configuration
        self.exec_cfg = exec_cfg      # Execution configuration
        self.path_root = path_root    # Root path

        # Data Configuration
        self.cal = None
        self.date_range = []
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())    # Item master
        self.cal_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())      # Calendar master
        self.input_col = 'discount'
        self.date_col = self.common['date_col']        # Date column name
        self.target_col = self.common['target_col']    # Target column name
        self.lag = lag
        self.hrchy_lvl = 6
        self.tb_name = 'M4S_I110521'

    def run(self):
        # Load the dataset about simulation
        exec_info_dict, exec_list = self.load_data()

        for info in exec_info_dict:
            self.io.update_from_db(sql=self.sql_conf.update_what_if_exec_info(**info))

        for sim in exec_list:    # For each simulation
            try:
                # What-if simulation
                result = self.simulate(sim=sim)

                # Convert simulation result to DB format
                result, info = self.make_db_format(result=result, sim=sim)

                if self.exec_cfg['save_db_yn']:
                    self.save_result(result=result, info=info)

            except ValueError:
                print("SP1 + SKU does not exist")

    # Load dataset
    def load_data(self):
        exec_info = self.io.get_df_from_db(sql=self.sql_conf.sql_what_if_exec_info())
        exec_info_dict = exec_info.to_dict('records')
        exec_df = self.io.get_df_from_db(sql=self.sql_conf.sql_what_if_exec_list())
        exec_list = exec_df.to_dict('records')

        return exec_info_dict, exec_list

    def simulate(self, sim: dict):
        self.set_date_range(sim=sim)

        # Data preprocessing
        pred, sales = self.preprocessing(sim=sim)

        # Add lagging data set
        pred = self.lagging(data=pred)

        # Add previous sales
        data = self.add_prev_sales(pred=pred, sales=sales)

        # Prediction
        result = self.predict(data=data, sim=sim)

        return result

    # Set date range
    def set_date_range(self, sim: dict):
        cal = self.cal_mst
        cal = cal[(cal['yy'] == sim['yy']) & (cal['week'] == sim['week'])]
        cal = cal[['start_week_day', 'end_week_day']].drop_duplicates().iloc[0].to_dict()

        date_list = list(pd.date_range(start=cal['start_week_day'], end=cal['end_week_day'], freq='W'))
        date_list = [datetime.strftime(date - timedelta(days=6), '%Y%m%d') for date in date_list]

        self.date_range = date_list

    # Data preprocess
    def preprocessing(self, sim: dict):
        # Load prediction data
        pred = self.load_prediction(sim=sim)

        # Slice the data set
        pred = pred[pred[self.date_col].isin(self.date_range)]

        # Add discount data
        pred['discount'] = sim['discount']
        # pred = pd.merge(pred, discount, on=self.date_col)

        self.cal = pred[[self.date_col, 'week']]

        # Change datetime type (String -> Datetime)
        pred[self.date_col] = pd.to_datetime(pred[self.date_col], format='%Y%m%d')
        pred[self.date_col] = pred[self.date_col].dt.date_sales

        # Get first day
        date_sorted = pred[self.date_col].sort_values()
        first_day = date_sorted.values[0]

        # Get days of previous week
        date_dict = self.get_prev_week_days(date=first_day)

        # Load the sales data
        sales = self.load_sales(date=date_dict, sim=sim)

        if len(sales) > 0:
            sales = sales['qty_lag'].sum()
        else:
            sales = 0

        # Set date to index
        pred = pred.set_index(keys=self.date_col)

        return pred, sales

    # Add previous sales
    @staticmethod
    def add_prev_sales(pred: pd.DataFrame, sales: float):
        pred.loc[:, 'qty_lag'][0] = sales

        return pred

    # Predict
    def predict(self, data, sim: dict):
        # Load best estimator
        estimator_info = self.load_best_estimator(sim=sim)
        estimator = estimator_info[3]

        # Predict
        result = estimator.predict(data)
        result = np.round(result, 2)    # round result

        return result

    # Load best algorithm
    def load_best_estimator(self, sim: dict):
        path = os.path.join(self.path_root, sim['data_vrsn_cd'], sim['division_cd'] + '_' +sim['data_vrsn_cd'] +
                            '_' + sim['sales_mgmt_cd'][-4:] + '-' + sim['item_cd'] + '.pickle')
        f = open(path, 'rb')
        estimator = pickle.load(f)

        return estimator

    def load_prediction(self, sim: dict):
        pred_conf = {
            'data_vrsn_cd': sim['data_vrsn_cd'],
            'division_cd': sim['division_cd'].upper(),
            'fkey': self.fkey_map[self.hrchy_lvl],
            'cust_grp_cd': sim['sales_mgmt_cd'][-4:],
            'item_cd': sim['item_cd']
        }
        prediction = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_item(**pred_conf))

        return prediction

    def load_sales(self, date: dict, sim: dict):
        sales_conf = {
            'division_cd': sim['division_cd'],
            'cust_grp_cd': sim['sales_mgmt_cd'][-4:],
            'item_cd': sim['item_cd'],
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

    # Get days of previous week
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

    def save_result(self, result: pd.DataFrame, info: dict) -> None:
        if self.exec_cfg['save_db_yn']:
            self.io.delete_from_db(sql=self.sql_conf.del_sim_result(**info))
            self.io.insert_to_db(df=result, tb_name=self.tb_name)

    def make_db_format(self, result: list, sim: dict):
        project_cd = self.common['project_cd']
        data_vrsn_cd = sim['data_vrsn_cd']

        df = pd.DataFrame({'result_sales': result})
        df['project_cd'] = project_cd
        df['data_vrsn_cd'] = data_vrsn_cd
        df['division_cd'] = sim['division_cd']
        df['wi_vrsn_id'] = sim['wi_vrsn_id']
        df['wi_vrsn_seq'] = sim['wi_vrsn_seq']
        df['sales_mgmt_cd'] = sim['sales_mgmt_cd']
        df['cust_grp_cd'] = sim['sales_mgmt_cd'][-4:]
        df['item_cd'] = sim['item_cd']
        df['yy'] = sim['yy']
        df['yymm'] = sim['yymm']
        df['week'] = sim['week']
        df['discount'] = sim['discount'] * 100
        df['create_user_cd'] = sim['create_user_cd']

        #
        # item_mst = self.item_mst.rename(columns=self.hrchy_cd_to_db_cd_map)
        # item_mst = item_mst.rename(columns=self.hrchy_sku_to_db_sku_map)
        # df = pd.merge(df, item_mst, how='left', on='item_cd')

        info = {
            'project_cd': project_cd,
            'data_vrsn_cd': data_vrsn_cd,
            'division_cd': sim['division_cd'],
            'wi_vrsn_id': sim['wi_vrsn_id'],
            'wi_vrsn_seq': sim['wi_vrsn_seq'],
            'sales_mgmt_cd': sim['sales_mgmt_cd'],
            'item_cd': sim['item_cd'],
            'yy': sim['yy'],
            'week': sim ['week'],
            'create_user_cd': sim['create_user_cd'],
        }

        return df, info
