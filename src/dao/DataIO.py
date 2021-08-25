from dao.SqlSession import SqlSession
from dao.SqlConfig import SqlConfig

import pickle
import pandas as pd


class DataIO(object):
    def __init__(self):
        self.sql_conf = SqlConfig()
        self.session = SqlSession()
        self.session.init()

    def get_date_range(self):
        date_from = self.session.select(sql=self.sql_conf.sql_comm_master(option='RST_START_DAY')).values[0][0]
        date_to = self.session.select(sql=self.sql_conf.sql_comm_master(option='RST_END_DAY')).values[0][0]

        return date_from, date_to

    def get_comm_info(self):
        common = self.session.select(sql=self.sql_conf.sql_comm_master())
        common['OPTION_CD'] = common['OPTION_CD'].apply(str.lower)
        common = common.set_index(keys='OPTION_CD').to_dict()['OPTION_VAL']

        return common

    def get_sell_in(self, date_from: int, date_to: int):
        sell_in = self.session.select(sql=self.sql_conf.sql_sell_in(date_from=date_from, date_to=date_to))

        return sell_in

    def get_sell_out(self, date_from: int, date_to: int):
        sell_out = self.session.select(sql=self.sql_conf.sql_sell_out(date_from=date_from, date_to=date_to))

        return sell_out

    @staticmethod
    def save_object(data, file_path: str, kind: str):
        if kind == 'csv':
            data.to_csv(file_path, )

        elif kind == 'binary':
            with open(file_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Data is saved")

    @staticmethod
    def load_object(file_path: str, kind: str):
        data = None
        if kind == 'csv':
            data = pd.read_csv(file_path)

        elif kind == 'binary':
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)

        return data
