import config
from SqlSession import SqlSession
from SqlConfig import SqlConfig

import numpy as np
import pandas as pd
from typing import List, Tuple


class DataPrep(object):
    COL_DROP_SELL = ['division_cd', 'seq']
    COL_DATETIME = ['yymmdd']
    COL_TARGET = ['']


    def __init__(self):
        # Path
        self.base_dir = config.BASE_DIR
        self.save_dir = config.SAVE_DIR

        # Condition
        self.variable_type = config.VAR_TYPE
        self.time_rule = config.RESAMPLE_RULE

        # Dataset
        self.data_preped = None

        # Hierarchy
        # self.hrchy: List[Tuple[int, str]] = [(1, 'chnl_cd'), (2, 'cust_cd'),  (3, 'pd_nm')]
        self.hrchy: List[Tuple[int, str]] = [(1, 'cust_grp'), (2, 'biz_cd'), (3, 'line_cd'),
                                             (4, 'brand_Cd'), (5, 'item_ctgr_cd')]
        self.hrchy_level = len(self.hrchy) - 1

        # Smoothing
        self.smooth_yn = config.SMOOTH_YN
        self.smooth_method = config.SMOOTH_METHOD
        self.smooth_rate = config.SMOOTH_RATE

    def preprocess(self, data: pd.DataFrame) -> None:
        print("Implement data preprocessing")

        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Grouping
        data_group = self.group(data=data)

        # Univariate or Multivariate dataset
        data_group = self.set_features(df=data_group)

        # resampling
        resampled_data = self.resample(df=data_group)

        self.data_preped = resampled_data

        print("Data preprocessing is finished\n")

    @ staticmethod
    def correct_target(df: pd.DataFrame) -> pd.DataFrame:
        df[config.COL_TARGET] = np.round(df['sales'] / df['store_price'])

        return df

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert columns to lower case
        df.columns = [col.lower() for col in df.columns]

        # drop unnecessary columns
        df = df.drop(columns=self.__class__.COL_DROP_SELL)

        # convert date column to datetime
        df[self.__class__.COL_DATETIME] = pd.to_datetime(df[self.__class__.COL_DATETIME], format='%Y%m%d')

        # convert target data type to float
        df[config.COL_TARGET] = df[config.COL_TARGET].astype(float)

        # # convert string type to int type
        # for col in self.__class__.COL_TYPE_NUM:
        #     if col in df.columns and df[col].dtype != int:
        #         df[col] = df[col].astype(int)

        # add noise feature
        # if config.ADD_EXO_YN:
        #     df = self.add_noise_feat(df=df)

        return df

    def group(self, data, cd=None, lvl=0) -> dict:
        grp = {}
        col = self.hrchy[lvl][1]

        code_list = None
        if isinstance(data, pd.DataFrame):
            code_list = list(data[col].unique())

        elif isinstance(data, dict):
            code_list = list(data[cd][col].unique())

        if lvl < self.hrchy_level:
            for code in code_list:
                sliced = None
                if isinstance(data, pd.DataFrame):
                    sliced = data[data[col] == code]
                elif isinstance(data, dict):
                    sliced = data[cd][data[cd][col] == code]
                result = self.group(data={code: sliced}, cd=code, lvl=lvl + 1)
                grp[code] = result

        elif lvl == self.hrchy_level:
            temp = {}
            for code in code_list:
                sliced = None
                if isinstance(data, pd.DataFrame):
                    sliced = data[data[col] == code]
                elif isinstance(data, dict):
                    sliced = data[cd][data[cd][col] == code]
                temp[code] = sliced

            return temp

        return grp

    def set_features(self, df=None, val=None, lvl=0) -> dict:
        temp = None
        if lvl == 0:
            temp = {}
            for key, val in df.items():
                result = self.set_features(val=val, lvl=lvl+1)
                temp[key] = result

        elif lvl < self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                result = self.set_features(val=val_hrchy, lvl=lvl+1)
                temp[key_hrchy] = result

            return temp

        elif lvl == self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                val_hrchy = val_hrchy[config.COL_TOTAL[self.variable_type]]
                temp[key_hrchy] = val_hrchy

            return temp

        return temp

    def resample(self, df=None, val=None, lvl=0):
        temp = None
        if lvl == 0:
            temp = {}
            for key, val in df.items():
                result = self.resample(val=val, lvl=lvl+1)
                temp[key] = result

        elif lvl < self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                result = self.resample(val=val_hrchy, lvl=lvl+1)
                temp[key_hrchy] = result

            return temp

        elif lvl == self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                val_hrchy = val_hrchy.set_index(config.COL_DATETIME)
                val_hrchy = val_hrchy.resample(rule=self.time_rule).sum()
                if self.smooth_yn:
                    val_hrchy = self.smoothing(df=val_hrchy)
                temp[key_hrchy] = val_hrchy

            return temp

        return temp

    @staticmethod
    def add_noise_feat(df: pd.DataFrame) -> pd.DataFrame:
        vals = df[config.COL_TARGET].values * 0.05
        vals = vals.astype(int)
        vals = np.where(vals == 0, 1, vals)
        vals = np.where(vals < 0, vals * -1, vals)
        noise = np.random.randint(-vals, vals)
        df['exo'] = df[config.COL_TARGET].values + noise

        return df

    def smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        for i, col in enumerate(df.columns):
            min_val = 0
            max_val = 0
            if self.smooth_method == 'quantile':
                min_val = df[col].quantile(self.smooth_rate)
                max_val = df[col].quantile(1 - self.smooth_rate)
            elif self.smooth_method == 'sigma':
                mean = np.mean(df[col].values)
                std = np.std(df[col].values)
                min_val = mean - 2 * std
                max_val = mean + 2 * std

            df[col] = np.where(df[col].values < min_val, min_val, df[col].values)
            df[col] = np.where(df[col].values > max_val, max_val, df[col].values)

        return df

    @staticmethod
    def split_sequence(df, n_steps_in, n_steps_out) -> tuple:
        """
        Split univariate sequence data
        :param df: Time series data
        :param n_steps_in:
        :param n_steps_out:
        :return:
        """
        data = df.astype('float32')
        x = []
        y = []
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(df):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, :]
            x.append(seq_x)
            y.append(seq_y)

        return np.array(x), np.array(y)