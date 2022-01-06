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

    def insert_to_db(self, df: pd.DataFrame, tb_name: str) -> None:
        self.session.insert(df=df, tb_name=tb_name)

    def delete_from_db(self, sql: str):
        self.session.delete(sql=sql)

    def update_from_db(self, sql: str):
        self.session.update(sql=sql)

    @staticmethod
    def save_object(data, data_type: str, file_path: str) -> None:
        """
        :param data
        :param data_type: csv / binary
        :param file_path: file path
        """
        if data_type == 'csv':
            data.to_csv(file_path, index=False)

        elif data_type == 'binary':
            with open(file_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Data is saved\n")

    @staticmethod
    def load_object(file_path: str, data_type: str):
        data = None
        if data_type == 'csv':
            data = pd.read_csv(file_path)
            # data = pd.read_csv(file_path, encoding='cp949')

        elif data_type == 'binary':
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)

        return data
