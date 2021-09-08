from baseline.analysis.Decomposition import Decomposition
import common.config as config
import common.util as util

import numpy as np
import pandas as pd


class DataPrep(object):
    DROP_COLS_DATA_PREP = ['division_cd', 'seq', 'from_dc_cd', 'unit_price', 'create_date']
    TYPE_STR_COLS = ['cust_cd', 'sku_cd']

    def __init__(self, date: dict, division: str, hrchy: list, decompose_yn=False):
        # Path
        self.base_dir = config.BASE_DIR
        self.save_dir = config.SAVE_DIR

        # Dataset
        self.division = division
        self.features = []
        self.target_col = config.TARGET_COL
        self.decompose_yn = decompose_yn
        self.resample_rule = config.RESAMPLE_RULE
        self.date_range = pd.date_range(start=date['date_from'],
                                        end=date['date_to'],
                                        freq=config.RESAMPLE_RULE)

        # Hierarchy
        self.hrchy = hrchy
        self.hrchy_level = len(hrchy) - 1

        # Smoothing
        self.smooth_yn = config.SMOOTH_YN
        self.smooth_method = config.SMOOTH_METHOD
        self.smooth_rate = config.SMOOTH_RATE

    def preprocess(self, data: pd.DataFrame) -> dict:
        print("Implement data preprocessing")
        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Grouping
        data_group = self.group(data=data)

        # Decomposition
        if self.decompose_yn:
            decompose = Decomposition(division=self.division,
                                      hrchy_list=self.hrchy,
                                      hrchy_lvl_cd=self.hrchy[self.hrchy_level])

            util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                 fn=decompose.decompose,
                                 df=data_group)

        # Resampling
        data_resample = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                             fn=self.resample,
                                             df=data_group)

        print("Data preprocessing is finished\n")

        return data_resample

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop unnecessary columns
        df = df.drop(columns=self.__class__.DROP_COLS_DATA_PREP, errors='ignore')

        #
        conditions = [df['unit_cd'] == 'EA ',
                      df['unit_cd'] == 'BOL',
                      df['unit_cd'] == 'BOX']

        values = [df['box_ea'], df['box_bol'], 1]
        unit_map = np.select(conditions, values)
        df['qty'] = df['qty'].to_numpy() / unit_map

        df = df.drop(columns=['box_ea', 'box_bol'], errors='ignore')

        # convert data type
        for col in self.TYPE_STR_COLS:
            df[col] = df[col].astype(str)

        # convert to datetime
        df['yymmdd'] = pd.to_datetime(df['yymmdd'], format='%Y%m%d')
        df = df.set_index(keys=['yymmdd'])

        # add noise feature
        # df = self.add_noise_feat(df=df)

        return df

    def group(self, data, cd=None, lvl=0) -> dict:
        grp = {}
        col = self.hrchy[lvl]

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

    def resample(self, df):
        cols = self.hrchy[:self.hrchy_level + 1]
        data_level = df[cols].iloc[0].to_dict()
        df_resampled = df.resample(rule=self.resample_rule).sum()

        # Check and add dates when sales does not exist
        if len(df_resampled.index) != len(self.date_range):
            idx_add = list(set(self.date_range) - set(df_resampled.index))
            data_add = np.zeros((len(idx_add), df_resampled.shape[1]))
            df_add = pd.DataFrame(data_add, index=idx_add, columns=df_resampled.columns)
            df_resampled = df_resampled.append(df_add)
            df_resampled = df_resampled.sort_index()

        data_lvl = pd.DataFrame(data_level, index=df_resampled.index)
        df_resampled = pd.concat([df_resampled, data_lvl], axis=1)

        return df_resampled

    def set_features(self, df):
        return df[self.features]

    def add_noise_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        vals = df[self.target_col].values * 0.05
        vals = vals.astype(int)
        vals = np.where(vals == 0, 1, vals)
        vals = np.where(vals < 0, vals * -1, vals)
        noise = np.random.randint(-vals, vals)
        df['exo'] = df[self.target_col].values + noise

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
