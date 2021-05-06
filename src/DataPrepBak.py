import config

from collections import defaultdict
import pandas as pd
import numpy as np


class DataPrep(object):
    COL_DATETIME = 'dt'
    COL_DROP_SELL = ['pd_cd']
    COL_VARIABLE = {'univ': ['dt', 'sales'],
                    'multi':  ['dt', 'sales', 'amt']}
    COL_EXO = ['dc']
    COL_TYPE_NUM = ['sales', 'amt', 'unit_price', 'store_price']
    COL_TYPE_POS = ['sales', 'amt', 'unit_price', 'store_price']

    def __init__(self):
        # Path
        self.base_dir = config.BASE_DIR
        self.sell_in_dir = config.SELL_IN_DIR
        self.sell_out_dir = config.SELL_OUT_DIR

        # Condition
        self.variable_type = config.VAR_TYPE
        self.group_type = config.GROUP_TYPE
        self.time_rule = config.RESAMPLE_RULE

        # Dataset
        self.sell_in_prep = None
        self.sell_out_prep = None

        # run data preprocessing
        self.preprocess()

    def preprocess(self) -> None:
        print("Implement data preprocessing")
        # Sell in dataset

        # load dataset
        sell_in = pd.read_csv(self.sell_in_dir, delimiter='\t', thousands=',')
        sell_out = pd.read_csv(self.sell_out_dir, delimiter='\t', thousands=',')

        # preprocess sales dataset
        sell_in = self.prep_sales(df=sell_in)
        sell_out = self.prep_sales(df=sell_out)

        # Grouping
        sell_in_group = self.group(df=sell_in)
        sell_out_group = self.group(df=sell_out)

        # Univariate or Multivariate dataset
        sell_in_group = self.set_features(df_group=sell_in_group)
        sell_out_group = self.set_features(df_group=sell_out_group)

        # resampling
        resampled_sell_in = self.resample(df_group=sell_in_group)
        resampled_sell_out = self.resample(df_group=sell_out_group)

        self.sell_in_prep = resampled_sell_in
        self.sell_out_prep = resampled_sell_out

        print("Data preprocessing is finished")

    def prep_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert columns to lower case
        df.columns = [col.lower() for col in df.columns]

        # drop unnecessary columns
        df = df.drop(columns=self.__class__.COL_DROP_SELL)

        # convert date column to datetime
        df[self.__class__.COL_DATETIME] = pd.to_datetime(df[self.__class__.COL_DATETIME], format='%Y%m%d')

        # remove ',' from numbers and
        # for col in self.__class__.COL_TYPE_NUM:
        #     if col in df.columns and df[col].dtype != int:
        #         df[col] = df[col].str.replace(',', '')

        # fill NaN
        is_null_col = [col for col, is_null in zip(df.columns, df.isnull().sum()) if is_null > 0]
        for col in is_null_col:
            df[col] = df[col].fillna(0)

        # convert string type to int type
        for col in self.__class__.COL_TYPE_NUM:
            if col in df.columns and df[col].dtype != int:
                df[col] = df[col].astype(int)

        # Filter minus values from dataset
        for col in self.__class__.COL_TYPE_POS:
            if col in df.columns:
                df = df[df[col] >= 0].reset_index(drop=True)

        return df

    def group(self, df: pd.DataFrame) -> dict:
        df_group = defaultdict(dict)
        for group_type in self.group_type:
            if group_type == 'all':
                df_group[group_type].update({group_type: df})
            elif group_type == 'pd':
                pd_types = df['pd_nm'].unique()
                for pd_type in pd_types:
                    df_group[group_type].update({pd_type: df[df['pd_nm'] == pd_type]})
            elif group_type == 'cust':
                cust_types = df['cust_cd'].unique()
                for cust_type in cust_types:
                    df_group[group_type].update({cust_type: df[df['cust_cd'] == cust_type]})

        return df_group

    def set_features(self, df_group: dict) -> dict:
        for group in df_group.values():
            for key, val in group.items():
                val = val[self.__class__.COL_VARIABLE[self.variable_type]]
                group[key] = val

        return df_group

    def resample(self, df_group: dict) -> dict:
        """
        Data Resampling
        :param df: time series dataset
        :return:
        """
        for group in df_group.values():
            for key, val in group.items():
                resampled = defaultdict(dict)
                for rule in self.time_rule:
                    val_dt = val.set_index(self.__class__.COL_DATETIME)
                    val_dt = val_dt.resample(rule=rule).sum()
                    resampled[rule] = val_dt
                group[key] = resampled

        return df_group

    @staticmethod
    def split_sequence_univ(df: pd.DataFrame, feature: str, timesteps: int, pred_steps=1):
        """
        Split univariate sequence data
        :param df: Time series data
        :param timesteps:
        :return:
        """
        data = df[feature].values
        n = len(data)

        X, y = list(), list()
        for i in range(n):
            # find the end of this pattern
            end_ix = i + timesteps
            pred_ix = end_ix + pred_steps
            # check if we are beyond the sequence
            if end_ix > n - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix], data[end_ix:pred_ix]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)