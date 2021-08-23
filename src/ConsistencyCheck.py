from SqlSession import SqlSession

import pandas as pd


class ConsistencyCheck(object):
    def __init__(self, division: str):
        self.division = division

    def check(self, df: pd.DataFrame):
        self.check_nan_data(df=df)
        self.check_code_map(df=df)

    def check_nan_data(self, df: pd.DataFrame):
        pass

    def check_code_map(self, df: pd.DataFrame):
        pass

    def check_data_type(self, df: pd.DataFrame):
        pass

    def check_discount(self, df: pd.DataFrame, col_nm: str):
        pass

    def fill_na(self, df: pd.DataFrame):
        is_null_col = [col for col, is_null in zip(df.columns, df.isnull().sum()) if is_null > 0]
        for col in is_null_col:
            df[col] = df[col].fillna(0)

    def save_result(self):
        pass