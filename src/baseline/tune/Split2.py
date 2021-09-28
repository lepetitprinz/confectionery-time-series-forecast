import common.config as config

import os
import numpy as np
import pandas as pd


class Split2(object):
    def __init__(self, data_vrsn_cd: str, division_cd: str, lvl: dict):
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division_cd

        # Data Level Configuration
        self.fixed_col = ['data_vrsn_cd', 'division_cd', 'stat_cd', 'week', 'yymmdd']
        self.hrchy_list = config.LVL_CD_LIST
        self.target_col = 'qty'

        self.ratio_lvl = lvl['lvl_ratio']
        self.split_lvl = lvl['lvl_split']

        self.ratio_hrchy = config.LVL_CD_LIST[:lvl['lvl_ratio']]
        self.ratio_cd = config.LVL_MAP[lvl['lvl_ratio']]
        self.ratio = None

        # Split
        self.split_hrchy = config.LVL_CD_LIST[:lvl['lvl_split']]

    def run(self, df_split, df_ratio):

        return None

    # Step 0. Filter unnecessary columns
    def filter_col(self, df: pd.DataFrame, kind: str):
        if kind == 'ratio':
            cols = self.fixed_col + self.hrchy_list[:self.ratio_lvl] + [self.target_col]
        elif kind == 'split':
            cols = self.fixed_col + self.hrchy_list[:self.split_lvl] + [self.target_col]

        return df[cols]

    # Step 1. Group by lower level quantity
    def group_by_agg(self, df: pd.DataFrame, group_lvl: int):
        col_group = self.fixed_col + self.hrchy_list[:group_lvl]
        df_agg = df.groupby(by=col_group).sum()
        df_agg = df_agg.reset_index()

        return df_agg

    def merge(self, left, right):
        on = list(left.columns)
        on.remove(self.target_col + '_' + 'upper')
        merged = pd.merge(left, right, on=on)

        return merged

    def rename_col(self, df: pd.DataFrame, lvl: str):
        df = df.rename(columns={self.target_col: self.target_col + '_' + lvl})
        return df

    def drop_qty(self, df):
        return df.drop(columns=['qty_lower', 'qty_upper'])

    def split_qty(self, upper, lower):
        on = list(upper.columns)
        on.remove(self.target_col)
        data = pd.merge(upper, lower, on=on)
        data['qty_split'] = data['qty'] / data['ratio']

        return data