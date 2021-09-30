from baseline.analysis.Decomposition import Decomposition
import common.util as util

import numpy as np
import pandas as pd
from collections import defaultdict


class DataPrep(object):
    DROP_COLS_DATA_PREP = ['division_cd', 'seq', 'from_dc_cd', 'unit_price', 'create_date']
    STR_TYPE_COLS = ['cust_cd', 'sku_cd']

    def __init__(self, date: dict, cust: pd.DataFrame, division: str, common: dict,
                 hrchy: list, decompose_yn=False):
        # Dataset configuration
        self.division = division
        self.cust = cust
        self.target_col = common['target_col']
        self.col_agg_map = {'sum': ['qty'],
                            'avg': ['discount', 'gsr_sum', 'rhm_avg', 'temp_avg', 'temp_max', 'temp_min']}
        self.seq_to_cust_map = {}
        self.resample_rule = common['resample_rule']
        self.date_range = pd.date_range(start=date['date_from'],
                                        end=date['date_to'],
                                        freq=common['resample_rule'])

        # Hierarchy configuration
        self.hrchy = hrchy
        self.hrchy_level = len(hrchy) - 1

        # Save & Load configuration
        self.decompose_yn = decompose_yn

    def preprocess(self, data: pd.DataFrame, exg: dict) -> dict:
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # convert data type
        for col in self.STR_TYPE_COLS:
            data[col] = data[col].astype(str)

        # Mapping: cust_cd -> cust_grp_cd
        data = pd.merge(data, self.cust, on=['cust_cd'], how='left')
        data['cust_grp_cd'] = data['cust_grp_cd'].fillna('-')

        # ------------------------------- #
        # 2. Preprocess Exogenous dataset
        # ------------------------------- #
        exg_all = util.prep_exg_all(data=exg['all'])

        # preprocess exogenous(partial) data
        exg_partial = util.prep_exg_partial(data=exg['partial'])

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Merge sales data & exogenous(all) data
        data = pd.merge(data, exg_all, on='yymmdd', how='left')

        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Grouping
        # data_group = self.group(data=data)
        data_group = util.group(hrchy=self.hrchy, hrchy_lvl=self.hrchy_level, data=data)

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

    def resample(self, df: pd.DataFrame):
        # Split by aggregation method
        df_sum = df[self.col_agg_map['sum']]
        df_avg = df[self.col_agg_map['avg']]

        # resampling
        df_sum_resampled = df_sum.resample(rule=self.resample_rule).sum()
        df_avg_resampled = df_avg.resample(rule=self.resample_rule).mean()

        # fill NaN
        df_sum_resampled = df_sum_resampled.fillna(value=0)
        df_avg_resampled = df_avg_resampled.fillna(value=0)

        # Concatenate aggregation
        df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        if len(df_resampled.index) != len(self.date_range):
            idx_add = list(set(self.date_range) - set(df_resampled.index))
            data_add = np.zeros((len(idx_add), df_resampled.shape[1]))
            df_add = pd.DataFrame(data_add, index=idx_add, columns=df_resampled.columns)
            df_resampled = df_resampled.append(df_add)
            df_resampled = df_resampled.sort_index()

        cols = self.hrchy[:self.hrchy_level + 1]
        data_level = df[cols].iloc[0].to_dict()
        data_lvl = pd.DataFrame(data_level, index=df_resampled.index)
        df_resampled = pd.concat([df_resampled, data_lvl], axis=1)

        return df_resampled

    @staticmethod
    def make_seq_to_cust_map(df: pd.DataFrame):
        seq_to_cust = df[['seq', 'cust_cd']].set_index('seq').to_dict('index')

        return seq_to_cust

    # def add_noise_feat(self, df: pd.DataFrame) -> pd.DataFrame:
    #     vals = df[self.target_col].values * 0.05
    #     vals = vals.astype(int)
    #     vals = np.where(vals == 0, 1, vals)
    #     vals = np.where(vals < 0, vals * -1, vals)
    #     noise = np.random.randint(-vals, vals)
    #     df['exo'] = df[self.target_col].values + noise
    #
    #     return df

    # def smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
    #     for i, col in enumerate(df.columns):
    #         min_val = 0
    #         max_val = 0
    #         if self.smooth_method == 'quantile':
    #             min_val = df[col].quantile(self.smooth_rate)
    #             max_val = df[col].quantile(1 - self.smooth_rate)
    #         elif self.smooth_method == 'sigma':
    #             mean = np.mean(df[col].values)
    #             std = np.std(df[col].values)
    #             min_val = mean - 2 * std
    #             max_val = mean + 2 * std
    #
    #         df[col] = np.where(df[col].values < min_val, min_val, df[col].values)
    #         df[col] = np.where(df[col].values > max_val, max_val, df[col].values)
    #
    #     return df
