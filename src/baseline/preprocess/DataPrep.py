from baseline.analysis.Decomposition import Decomposition
import common.config as config
import common.util as util

import numpy as np
import pandas as pd


class DataPrep(object):
    DROP_COLS_DATA_PREP = ['division_cd', 'seq', 'from_dc_cd', 'unit_price', 'create_date']
    GROUP_BY_COLS = ['week']
    # GROUP_BY_COLS = ['sold_cust_grp_cd', 'week']
    TYPE_STR_COLS = ['sold_cust_grp_cd', 'item_cd', 'yymmdd']
    TARGET_COL = ['qty']

    def __init__(self):
        # Path
        self.base_dir = config.BASE_DIR
        self.save_dir = config.SAVE_DIR

        # Dataset
        self.division = ''
        self.features = []

        # Hierarchy
        self.hrchy_list = config.HRCHY_LIST
        self.hrchy = config.HRCHY
        self.hrchy_level = config.HRCHY_LEVEL

        # Smoothing
        self.smooth_yn = config.SMOOTH_YN
        self.smooth_method = config.SMOOTH_METHOD
        self.smooth_rate = config.SMOOTH_RATE

    def preprocess(self, data: pd.DataFrame, division: str) -> dict:
        print("Implement data preprocessing")

        # set dataset division
        self.division = division

        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Grouping
        data_group = self.group(data=data)

        # Decomposition
        decompose = Decomposition(division=self.division,
                                  hrchy_list=self.hrchy_list,
                                  hrchy_lvl_cd=self.hrchy_list[self.hrchy_level])

        util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                             fn=decompose.decompose,
                             df=data_group)

        # Resampling
        data_group = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                          fn=self.resample,
                                          df=data_group)


        # Univariate or Multivariate dataset
        # data_featured = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
        #                                      fn=self.set_features,
        #                                      df=data_resampled)

        print("Data preprocessing is finished\n")

        return data_group

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert columns to lower case
        df.columns = [col.lower() for col in df.columns]

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

        # add noise feature
        # df = self.add_noise_feat(df=df)

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

    def resample(self, df):
        cols = self.GROUP_BY_COLS + self.hrchy_list[:self.hrchy_level+1]
        df_group = df.groupby(by=cols).sum()
        df_group = df_group.reset_index()

        return df_group

    def set_features(self, df):
        return df[self.features]

    def add_noise_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        vals = df[self.TARGET_COL].values * 0.05
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