import common.config as config

import os
import numpy as np
import pandas as pd


class Split(object):
    def __init__(self, data_vrsn_cd: str, division_cd: str, lvl: dict):
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division_cd

        # Data Level Configuration
        self.fixed_col = ['data_vrsn_cd', 'division_cd', 'stat_cd', 'week', 'yymmdd']
        self.hrchy_list = config.LVL_CD_LIST
        self.target_col = 'qty'

        self.ratio_lvl = lvl['lvl_ratio']
        self.split_lvl = lvl['lvl_split']

    def run(self, df_split, df_ratio):
        # Filter columns
        df_split_filtered = self.filter_col(df=df_split, kind='split')
        df_ratio_filtered = self.filter_col(df=df_ratio, kind='ratio')

        # acyclic iteration
        count = self.ratio_lvl - self.split_lvl
        ratio = self.ratio_iter(df_ratio=df_ratio_filtered)
        split = self.split_iter(dict_ratio=ratio, split=df_split_filtered)

        return split

    def ratio_iter(self, df_ratio: pd.DataFrame) -> dict:
        agg_dict = self.agg_data(df=df_ratio)

        ratio_dict = {}
        for i in range(self.split_lvl, self.ratio_lvl):
            upper = agg_dict[i]
            lower = agg_dict[i+1]
            upper = self.rename_col(df=upper, lvl='upper')
            lower = self.rename_col(df=lower, lvl='lower')
            ratio = self.calc_ratio(df_upper=upper, df_lower=lower)
            ratio_dict[i] = ratio

        return ratio_dict

    def agg_data(self, df: pd.DataFrame) -> dict:
        agg_dict = {self.ratio_lvl: df}
        for i in range(self.ratio_lvl, self.split_lvl, -1):
            ratio_grp = self.group_by_agg(df=df, group_lvl=i-1)
            agg_dict[i-1] = ratio_grp

        return agg_dict

    def split_iter(self, dict_ratio: dict, split: pd.DataFrame) -> pd.DataFrame:
        count = self.ratio_lvl - self.split_lvl
        lvl = self.split_lvl
        result = None
        while count != 0:
            result = self.split(dict_ratio=dict_ratio, split=split, lvl=lvl)
            lvl += 1
            count -= 1

        return result

    def split(self, dict_ratio: dict, split: pd.DataFrame, lvl: int) -> pd.DataFrame:
        ratio = dict_ratio[lvl]
        split_qty = self.split_qty(upper=split, lower=ratio)

        return split_qty

    # Step 0. Filter unnecessary columns
    def filter_col(self, df: pd.DataFrame, kind: str) -> pd.DataFrame:
        cols = None
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

    # Step2. Calculate ratio
    def calc_ratio(self, df_upper, df_lower):
        result = self.merge(left=df_upper, right=df_lower)
        result['ratio'] = result['qty_lower'] / result['qty_upper']
        result = self.drop_qty(df=result)

        return result

    def merge(self, left, right):
        on = list(left.columns)
        on.remove(self.target_col + '_' + 'upper')
        merged = pd.merge(left, right, on=on)

        return merged

    def rename_col(self, df: pd.DataFrame, lvl: str):
        df = df.rename(columns={self.target_col: self.target_col + '_' + lvl})
        return df

    @staticmethod
    def drop_qty(df):
        return df.drop(columns=['qty_lower', 'qty_upper'])

    def split_qty(self, upper, lower):
        on = list(upper.columns)
        on.remove(self.target_col)
        split = pd.merge(upper, lower, on=on)
        split['qty'] = split['qty'] * split['ratio']
        split = split.drop(columns='ratio')

        return split
