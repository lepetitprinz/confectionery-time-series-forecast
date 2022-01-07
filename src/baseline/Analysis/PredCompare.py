import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import datetime
import pandas as pd


class PredCompare(object):
    col_fixed = ['division_cd', 'start_week_day', 'week']

    def __init__(self, data_cfg: dict):
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

        # Data configuration
        self.data_cfg = data_cfg
        self.division = data_cfg['division']
        self.item_lvl = data_cfg['item_lvl']
        self.week_compare = 1
        self.date = {}
        self.level = {}
        self.hrchy = {}
        self.data_vrsn_cd = ''

    def run(self):
        self.init()
        sales, pred = self.load_dataset()
        sales = self.resample_sales(data=sales)
        result = self.merge_result(sales=sales, pred=pred)

    def init(self) -> None:
        self.set_data_version()    # Set data version
        self.set_date()    # Set date
        self.set_level(item_lvl=self.item_lvl)
        self.set_hrchy()

    def set_data_version(self) -> str:
        today = datetime.date.today()
        today = today - datetime.timedelta(days=today.weekday())
        hist_from = today - datetime.timedelta(days=int(self.common['week_hist']) * 7)
        hist_to = today - datetime.timedelta(days=1)

        # convert to string type
        hist_from = hist_from.strftime('%Y%m%d')
        hist_to = hist_to.strftime('%Y%m%d')

        return hist_from + '-' + hist_to

    def set_date(self) -> None:
        if self.data_cfg['cycle_yn']:
            self.date = self.set_compare_week()
            self.data_vrsn_cd = self.set_data_version()
        else:
            self.date = self.data_cfg['date']
            self.data_vrsn_cd = self.data_cfg['data_vrsn_cd']

    def set_level(self,  item_lvl: int):
        level = {
            'cust_lvl': 1,    # Fixed
            'item_lvl': item_lvl,
        }
        self.level = level

    def set_hrchy(self):
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

    def set_compare_week(self):
        today = datetime.date.today()
        today = today - datetime.timedelta(days=today.weekday())

        compare_from = today - datetime.timedelta(days=self.week_compare * 7 - 1)
        compare_to = today - datetime.timedelta(days=1)

        compare_from = compare_from.strftime('%Y%m%d')
        compare_to = compare_to.strftime('%Y%m%d')

        date = {'from': compare_from, 'to': compare_to}

        return date

    def load_dataset(self):
        # load sales
        info_sales = {
            'division_cd': self.division,
            'start_week_day': self.date['from']
        }
        sales = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week(**info_sales))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week(**info_sales))

        # load prediction
        info_pred = {
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'fkey': self.hrchy['key'][:-1],
            'yymmdd': self.date['from'],
        }
        pred = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_best(**info_pred))
        pred = self.filter_col(data=pred)

        return sales, pred

    def filter_col(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_col = self.hrchy['apply'] + self.col_fixed + ['pred']
        data = data[filter_col]

        return data

    def resample_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        grp_col = self.hrchy['apply'] + self.col_fixed
        data = data.groupby(by=grp_col).sum()
        data = data.reset_index()

        return data

    def merge_result(self, sales: pd.DataFrame, pred: pd.DataFrame):
        merge_col = self.hrchy['apply'] + self.col_fixed
        merged = pd.merge(
            sales,
            pred,
            on=merge_col,
            how='left'
        )

        return merged
