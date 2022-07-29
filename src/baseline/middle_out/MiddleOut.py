import common.util as util
import common.config as config
from baseline.feature_engineering.importance import FeatureImportance

import numpy as np
import pandas as pd
from copy import deepcopy


class MiddleOut(object):
    def __init__(
            self,
            division: str,
            data_vrsn: str,
            yy_week: pd.DataFrame,
            common: dict,
            hrchy: dict,
            ratio_lvl,
            item_mst: pd.DataFrame
    ):
        """
        :param division: Division (SELL-IN/SELl-OUT)
        :param data_vrsn: Data version code
        :param common: Common information
        :param hrchy: Hierarchy information
        :param ratio_lvl: Middle-out level
        :param item_mst: Several master information
        """
        # Data Information instance attribute
        self.common = common
        self.yy_week = yy_week
        self.division_cd = division
        self.data_vrsn_cd = data_vrsn

        # Data Level instance attribute
        self.hrchy = hrchy  # Hierarchy information
        self.hrchy_cust_cd_list = ['cust_grp_cd']
        self.hrchy_item_cd_list = common['db_hrchy_item_cd'].split(',')
        self.hrchy_item_nm_list = common['db_hrchy_item_nm'].split(',')
        self.hrchy_cd_to_db_cd = config.HRCHY_CD_TO_DB_CD_MAP
        self.drop_col = ['project_cd', 'division_cd', 'data_vrsn_cd', 'fkey', 'create_user_cd', 'accuracy'] + \
                        self.hrchy_item_nm_list

        # Weight instance attribute
        self.n_weight = 5
        self.apply_method = 'weight'    # all / weight

        # Middle-out instance attribute
        self.err_val = 0  # Setting value for prediction error
        self.max_val = 10 ** 5 - 1  # Clipping value for outlier
        self.ratio_lvl = ratio_lvl  # Ratio level
        self.target_col = 'sales'  # Target column
        self.item_mst = item_mst  # Item master
        self.split_lvl = self.hrchy['lvl']['item']

        # After processing instance attribute
        self.rm_special_char_list = ['item_attr03_nm', 'item_attr04_nm', 'item_nm']

    def run_middle_out(self, sales: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        # Convert uppercase columns to lower
        sales = self.col_to_lower(data=sales)
        pred = self.col_to_lower(data=pred)

        data_split = self.prep_split(data=pred)  # preprocess the prediction result

        # Calculate weights
        data_ratio = None
        if self.apply_method == 'all':
            data_ratio = self.prep_ratio_all(data=sales)

        elif self.apply_method == 'weight':
            # Feature Engineering : Sales
            importance = FeatureImportance(item_mst=self.item_mst, yy_week=self.yy_week, n_feature=self.n_weight)
            weights = importance.run(data=sales)
            data_ratio = self.prep_ratio_with_weight(data=sales, weight=weights)  # preprocess the recent sales history

        # apply middle-out
        middle_out = self.middle_out(data_split=data_split, data_ratio=data_ratio)    # Middle out
        middle_out_db = self.after_processing(data=middle_out)    # After process the middle out result

        return middle_out_db
        # return middle_out_db, middle_out

    # Preprocess the prediction result
    def prep_split(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=self.drop_col, errors='ignore')  # Drop unnecessary columns
        data = data.rename(columns={'result_sales': 'sales'})     # Rename column

        return data

    # Preprocess the recent sales history
    def prep_ratio_all(self, data: pd.DataFrame) -> pd.DataFrame:
        item_temp = deepcopy(self.item_mst)
        item_col = [col for col in item_temp.columns if 'nm' not in col]    # Item code list
        item_temp = item_temp[item_col]    # Filter item code data

        merged = pd.merge(data, item_temp, how='left', on=['sku_cd'])    # Merge item information
        ratio = self.agg_by_data_level(data_ratio=merged, item_col=item_col)
        ratio = ratio.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)    # Rename columns

        return ratio

    # Preprocess the recent sales history
    def prep_ratio_with_weight(self, data: pd.DataFrame, weight) -> pd.DataFrame:
        item_temp = deepcopy(self.item_mst)
        item_col = [col for col in item_temp.columns if 'nm' not in col]    # Item code list
        item_temp = item_temp[item_col]    # Filter item code data
        merged = pd.merge(data, item_temp, how='left', on=['sku_cd'])    # Merge item information

        # aggregate by each weight
        ratio = self.agg_by_weight(data=merged, weight=weight)
        ratio = ratio.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)    # Rename columns

        return ratio

    @staticmethod
    def agg_by_weight(data, weight):
        merged = pd.merge(data, weight, how='inner', on=['cust_grp_cd', 'brand_cd', 'week'])
        merged['weight_sales'] = merged['sales'] * merged['weight']

        agg_sales = merged.groupby(
            by=['cust_grp_cd', 'biz_cd', 'line_cd', 'brand_cd', 'item_cd', 'sku_cd']
        ).sum().reset_index()

        agg_sales = agg_sales.drop(columns=['weight', 'sales'])
        agg_sales = agg_sales.rename(columns={'weight_sales': 'sales'})

        return agg_sales

    # Preprocess the recent sales history
    def prep_ratio_bak(self, data: pd.DataFrame) -> pd.DataFrame:
        item_temp = deepcopy(self.item_mst)
        item_col = [col for col in item_temp.columns if 'nm' not in col]  # Item code list
        item_temp = item_temp[item_col]  # Filter item code data

        merged = pd.merge(data, item_temp, how='left', on=['sku_cd'])  # Merge item information
        ratio = self.agg_by_data_level(data_ratio=merged, item_col=item_col)
        ratio = ratio.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)  # Rename columns

        return ratio

    def middle_out(self, data_split: pd.DataFrame, data_ratio: pd.DataFrame) -> pd.DataFrame:
        # Acyclic iteration
        count = self.ratio_lvl - self.hrchy['lvl']['item']  # Level count
        ratio = self.ratio_iter(df_ratio=data_ratio)  #
        result = self.split_iter(dict_ratio=ratio, split=data_split)  #

        return result

    def after_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        item_mst = self.item_mst

        # Rename item master column
        item_mst.columns = [self.hrchy_cd_to_db_cd.get(col, col) for col in item_mst.columns]
        item_mst = item_mst.rename(columns={'sku_cd': 'item_cd', 'sku_nm': 'item_nm'})
        data = data.rename(columns={self.target_col: 'result_sales', 'sku_cd': 'item_cd'})

        # Merge data and item master
        merged = pd.merge(
            data,
            item_mst,
            how='left',
            on=self.hrchy_item_cd_list
        )
        # Add db information
        result = self.add_db_information(data=merged)

        # Remove Special Character (prevent insert process from value error)
        for col in self.rm_special_char_list:
            if col in list(result.columns):
                result = util.remove_special_character(data=result, feature=col)

        # convert 'inf' or '-inf' to zero
        result['result_sales'] = np.nan_to_num(result['result_sales'].values, posinf=0, neginf=0)

        return result

    @staticmethod
    def col_to_lower(data: pd.DataFrame) -> pd.DataFrame:
        data.columns = [col.lower() for col in list(data.columns)]

        return data

    @staticmethod
    def agg_by_data_level(data_ratio: pd.DataFrame, item_col: list) -> pd.DataFrame:
        agg_col = ['cust_grp_cd'] + item_col
        data_agg = data_ratio.groupby(by=agg_col).mean()
        data_agg = data_agg.reset_index()

        return data_agg

    def add_db_information(self, data: pd.DataFrame) -> pd.DataFrame:
        data['project_cd'] = self.common['project_cd']
        data['data_vrsn_cd'] = self.data_vrsn_cd
        data['division_cd'] = self.division_cd
        data['fkey'] = 'C1-P5' + '-MIDDLE-OUT-' + data['cust_grp_cd'] + '-' + data['item_cd']
        # data['fkey'] = self.hrchy['key'] + 'MIDDLE-OUT-' + data['cust_grp_cd'] + '-' + data['item_cd']

        return data

    def add_del_information(self) -> dict:
        info = {
            'project_cd': self.common['project_cd'],
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division_cd,
            'fkey': 'C1-P5'
        }

        return info

    # Group by low level quantity
    def group_by_agg(self, df: pd.DataFrame, group_lvl: int) -> pd.DataFrame:
        col_group = self.hrchy_cust_cd_list + self.hrchy_item_cd_list[:group_lvl]
        df_agg = df.groupby(by=col_group).sum()
        df_agg = df_agg.reset_index()

        return df_agg

    # Calculate ratio
    def calc_ratio(self, df_upper: pd.DataFrame, df_lower: pd.DataFrame) -> pd.DataFrame:
        result = self.merge_df(left=df_upper, right=df_lower)

        # calculate ratio of lower quantity divided by upper quantity
        result['ratio'] = result[self.target_col + '_' + 'lower'] / result[self.target_col + '_' + 'upper']

        # Convert inf or -inf to zeros
        result['ratio'] = np.nan_to_num(result['ratio'].values, posinf=0, neginf=0)

        # Remove sales quantity columns
        result = self.drop_qty(df=result)

        return result

    # Iterative calculation by each upper & lower level
    def ratio_iter(self, df_ratio: pd.DataFrame) -> dict:
        agg_dict = self.agg_data(df=df_ratio)

        ratio_dict = {}
        for i in range(self.split_lvl, self.ratio_lvl):
            upper = agg_dict[i]  # Set upper level data
            lower = agg_dict[i + 1]  # Set lower level data
            upper = self.rename_col(df=upper, lvl='upper')  # Rename target column (sales)
            lower = self.rename_col(df=lower, lvl='lower')  # Rename target column (sales)
            ratio = self.calc_ratio(df_upper=upper, df_lower=lower)  # calculate ratio of each level
            ratio_dict[i] = ratio

        return ratio_dict

    # Aggregate sales data
    def agg_data(self, df: pd.DataFrame) -> dict:
        agg_dict = {self.ratio_lvl: df}
        for i in range(self.ratio_lvl, self.split_lvl, -1):
            ratio_grp = self.group_by_agg(df=df, group_lvl=i - 1)
            agg_dict[i - 1] = ratio_grp

        return agg_dict

    def split_iter(self, dict_ratio: dict, split: pd.DataFrame) -> pd.DataFrame:
        count = self.ratio_lvl - self.split_lvl
        lvl = self.split_lvl
        while count != 0:
            split = self.split(dict_ratio=dict_ratio, split=split, lvl=lvl)
            lvl += 1
            count -= 1

        return split

    def split(self, dict_ratio: dict, split: pd.DataFrame, lvl: int) -> pd.DataFrame:
        ratio = dict_ratio[lvl]
        split_qty = self.split_qty(upper=split, lower=ratio, lvl=lvl)

        return split_qty

    def merge_df(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        on = list(left.columns)
        on.remove(self.target_col + '_' + 'upper')
        merged = pd.merge(left, right, on=on)

        return merged

    def rename_col(self, df: pd.DataFrame, lvl: str) -> pd.DataFrame:
        df = df.rename(columns={self.target_col: self.target_col + '_' + lvl})

        return df

    def drop_qty(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[self.target_col + '_' + 'lower', self.target_col + '_' + 'upper'])

    def split_qty(self, upper: pd.DataFrame, lower: pd.DataFrame, lvl: int) -> pd.DataFrame:
        on = self.hrchy_cust_cd_list + self.hrchy_item_cd_list[:lvl]
        split = pd.merge(upper, lower, on=on)
        split[self.target_col] = round(split[self.target_col] * split['ratio'], 2)

        # clip & round results
        split[self.target_col] = np.clip(split[self.target_col].values, 0, self.max_val)
        split[self.target_col] = np.round(split[self.target_col].values, 2)

        split = split.drop(columns='ratio')

        return split
