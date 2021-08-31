from common.SqlSession import SqlSession
from common.SqlConfig import SqlConfig

import pickle
import pandas as pd


class DataIO(object):
    def __init__(self):
        self.sql_conf = SqlConfig()
        self.session = SqlSession()
        self.session.init()

    def get_df_from_db(self, sql) -> pd.DataFrame:
        df = self.session.select(sql=sql)
        df.columns = [col.lower() for col in df.columns]
        return df

    def get_dict_from_db(self, sql, key, val) -> dict:
        df = self.session.select(sql=sql)
        df[key] = df[key].apply(str.lower)
        result = df.set_index(keys=key).to_dict()[val]

        return result

    def insert_to_db(self, df: pd.DataFrame, tb_name: str):
        self.session.insert(df=df, tb_name=tb_name)

    def update_to_db(self, df: pd.DataFrame, tb_name: str):
        self.session.update(df=df, tb_name=tb_name)

    @staticmethod
    def save_object(data, kind: str, file_path: str):
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
