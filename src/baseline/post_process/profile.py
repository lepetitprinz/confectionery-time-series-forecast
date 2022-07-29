from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, Dict, List


class Profile(object):
    def __init__(self, exec_cfg, apply_cfg, path, hist_to=''):
        self.io = None
        self.sql = None

        self.exec_cfg = exec_cfg
        self.apply_cfg = apply_cfg

        # Data instance attribute
        self.date = {}
        self.hrchy = {}
        self.path = path
        self.hist_to = hist_to
        self.division = 'SELL_IN'

    def run(self, sales: pd.DataFrame, acc: pd.DataFrame):
        # Initialize profiling setting
        self.init()

        # Load dataset
        sales, accuracy = self.load()

        # Preprocessing
        sales = self.make_hrchy_data(data=sales)

        # Calculate the coefficient of variant
        self.calc_cv(data=sales)

        # Filter the accuracy on accuracy threshold
        accuracy = self.filter_acc_threshold(data=accuracy)

        self.profile_item()

    def filter_acc_threshold(self, data: pd.DataFrame):
        data = data[(data['accuracy'] >= self.apply_cfg['acc_threshold']) &
                    (data['accuracy'] <= (2 - self.apply_cfg['acc_threshold']))]

        return data

    def init(self):
        # Instantiate data I/O class
        self.io = DataIO()
        self.sql = SqlConfig()

        # Set sales date range
        self.set_date_range()

    def set_date_range(self) -> None:
        if self.exec_cfg['batch']:
            self.set_batch_date()
        else:
            hist_to_datetime = dt.datetime.strptime(self.hist_to, '%Y%m%d')
            hist_from = hist_to_datetime - dt.timedelta(weeks=self.apply_cfg['weeks']) + dt.timedelta(days=1)
            hist_from = dt.datetime.strftime(hist_from, '%Y%m%d')

            self.date = {
                'from': hist_from,
                'to': self.hist_to
            }

    def set_batch_date(self):
        today = dt.date.today()
        tihs_monday = today - dt.timedelta(days=today.weekday())

        hist_from = tihs_monday - dt.timedelta(days=self.apply_cfg['weeks'] * 7)
        hist_to = tihs_monday - dt.timedelta(days=1)

        hist_from = hist_from.strftime('%Y%m%d')
        hist_to = hist_to.strftime('%Y%m%d')

        self.date = {
            'from': hist_from,
            'to': hist_to
        }

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sales = self.load_sales()
        accuracy = self.load_accuracy()

        return sales, accuracy

    def load_sales(self) -> pd.DataFrame:
        sales = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales = self.io.get_df_from_db(sql=self.sql.sql_sell_in_week_grp(**self.date))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales = self.io.get_df_from_db(sql=self.sql.sql_sell_out_week_grp(**self.date))

        return sales

    def load_accuracy(self):
        return pd.DataFrame()

    def calc_cv(self, data):
        pass

    def make_hrchy_data(self, data: pd.DataFrame):
        return data

    def profile_item(self):
        pass