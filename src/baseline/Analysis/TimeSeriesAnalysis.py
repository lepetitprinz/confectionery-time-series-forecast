import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import os
import pandas as pd
from typing import Tuple, Dict
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller


class TimeSeriesAnalysis(object):
    col_fixed = ['division_cd', 'start_week_day', 'week']

    def __init__(self, data_cfg: dict, exec_cfg: dict):
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.root_path = data_cfg['root_path']
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg

        # Data configuration
        self.division = data_cfg['division']
        self.data_version = data_cfg['data_version']
        self.date = {
            'from': self.data_version.split('-')[0],
            'to': self.data_version.split('-')[1]
        }
        self.target_col = data_cfg['target_col']

        self.level = {}
        self.hrchy = {}

    def run(self):
        self.init()

        sales, accuracy = self.load_dataset()

        sales, accuracy = self.preprocessing(sales=sales, accuracy=accuracy)

        self.analysis(sales=sales, accuracy=accuracy)

    def init(self):
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()

    def set_level(self,  item_lvl: int) -> None:
        level = {
            'cust_lvl': 1,    # Fixed
            'item_lvl': item_lvl,
        }
        self.level = level

    def set_hrchy(self) -> None:
        self.hrchy = {
            'cnt': 0,
            'key': "C" + str(self.level['cust_lvl']) + '-' + "P" + str(self.level['item_lvl']) + '-',
            'lvl': {
                'cust': self.level['cust_lvl'],
                'item': self.level['item_lvl'],
                'total': self.level['cust_lvl'] + self.level['item_lvl']
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),
                'item': [config.HRCHY_CD_TO_DB_CD_MAP.get(col, 'item_cd') for col
                         in self.common['hrchy_item'].split(',')[:self.level['item_lvl']]]
            }
        }
        self.hrchy['apply'] = self.hrchy['list']['cust'] + self.hrchy['list']['item']

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load sales
        sales = self.load_sales()

        # Load accuracy dataset
        path = os.path.join(self.root_path, self.data_version, 'result', self.data_version + '_' +
                            self.division + '_' + str(self.hrchy['lvl']['item']) + '.csv')
        accuracy = pd.read_csv(path, )

        return sales, accuracy

    def load_sales(self) -> pd.DataFrame:
        sales = None
        info_sales_hist = {
            'division_cd': self.division,
            'from': self.date['from'],
            'to': self.date['to']
        }
        if self.division == 'SELL_IN':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))

        elif self.division == 'SELL_OUT':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))

        return sales

    def preprocessing(self, sales: pd.DataFrame, accuracy: pd.DataFrame):
        sales = self.rename_column(data=sales)
        sales = self.resample_sales(data=sales)

        accuracy = self.conv_to_str_type(data=accuracy, col='cust_grp_cd')

        return sales, accuracy

    @staticmethod
    def rename_column(data: pd.DataFrame) -> pd.DataFrame:
        cols = [config.HRCHY_CD_TO_DB_CD_MAP.get(col, col) for col in data.columns]
        cols = [config.HRCHY_SKU_TO_DB_SKU_MAP.get(col, col) for col in cols]
        data.columns = cols

        return data

    def resample_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        grp_col = self.hrchy['apply'] + self.col_fixed
        data = data.groupby(by=grp_col).sum()
        data = data.reset_index()

        return data

    @staticmethod
    def conv_to_str_type(data: pd.DataFrame, col: str):
        data[col] = data[col].astype(str)

        return data

    def analysis(self, sales: pd.DataFrame, accuracy: pd.DataFrame):
        item_lvl_cd = self.hrchy['apply'][-1]
        data_level = sales[['cust_grp_cd', item_lvl_cd]].drop_duplicates()

        chk_st = []
        for cust, item in zip(data_level['cust_grp_cd'], data_level[item_lvl_cd]):
            temp = sales[(sales['cust_grp_cd'] == cust) & (sales[item_lvl_cd] == item)]
            temp = temp.sort_values(by=['start_week_day'])

            if len(temp) > 5:
                chk_st.append(self.check_stationarity(data=temp[self.target_col]))
            else:
                chk_st.append(False)

        print("")
        # Check stationarity

    def group_by_accuracy(self):
        pass

    def check_white_noise(self):
        pass

    def check_stationarity(self, data: pd.Series) -> bool:
        """
        Strong stationarity
            a stochastic process whose unconditional joint probability distribution does not change
            when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
        Weak stationarity
            a process where mean, variance, auto-correlation are constant throughout the time
        """
        result = self.augmented_dickey_fuller_test(data=data)

        return result

    @staticmethod
    def augmented_dickey_fuller_test(data: pd.Series) -> bool:
        stationary = False
        result = adfuller(data.values)
        if result[1] <= 0.05:
            stationary = True

        return stationary
