from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

from copy import deepcopy
import numpy as np
import pandas as pd


class ConsistencyCheck(object):
    def __init__(self, division: str, common: dict, hrchy: dict, date: dict,
                 err_grp_map: dict, save_yn: bool):
        # Class Configuration
        self.io = DataIO()
        self.sql_config = SqlConfig()

        # Data Configuration
        self.division = division
        self.common = common
        self.hrchy = hrchy
        self.data_vrsn_cd = date['date_from'] + '-' + date['date_to']
        self.err_grp_map = err_grp_map
        self.unit_cd = common['unit_cd'].split(',')

        # Save and Load Configuration
        self.cns_tb_name = 'M4S_I002174'
        self.save_yn = save_yn

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        # Code Mapping
        normal = self.check_code_map(df=df)

        return normal

    def check_code_map(self, df: pd.DataFrame) -> pd.DataFrame:
        normal = self.check_prod_level(df=df)
        if self.division == 'SELL_IN':
            normal = self.check_unit_code(df=normal)
            normal = self.check_unit_code_map(df=normal)

        return normal

    # Error 1
    def check_prod_level(self, df: pd.DataFrame):
        test_df = deepcopy(df)
        test_df = test_df[self.hrchy['apply']]
        na_rows = test_df.isna().sum(axis=1) > 0
        err = df[na_rows].fillna('')
        normal = df[~na_rows]

        # save the error data
        err = self.make_err_format(df=err, err_cd='err001')
        if len(err) > 0 and self.save_yn:
            self.io.insert_to_db(df=err, tb_name=self.cns_tb_name)

        return normal

    # Error 2
    def check_unit_code(self, df: pd.DataFrame):
        err = df[~df['unit_cd'].isin(self.unit_cd)]
        normal = df[df['unit_cd'].isin(self.unit_cd)]

        err = self.make_err_format(df=err, err_cd='err002')
        if len(err) > 0 and self.save_yn:
            self.io.insert_to_db(df=err, tb_name=self.cns_tb_name)

        return normal

    # Error 3
    def check_unit_code_map(self, df: pd.DataFrame):
        unit_code_map = self.io.get_df_from_db(sql=self.sql_config.sql_unit_map())
        unit_code_map.columns = [col.lower() for col in unit_code_map.columns]
        df['sku_cd'] = df['sku_cd'].astype(str)

        # Merge unit code map
        merged = pd.merge(df, unit_code_map, how='left', on='sku_cd')
        err = merged[merged['box_bol'].isna()]
        normal = merged[~merged['box_bol'].isna()]

        # Save Error
        err = self.make_err_format(df=err, err_cd='err003')
        if len(err) > 0 and self.save_yn:
            self.io.insert_to_db(df=err, tb_name=self.cns_tb_name)

        return normal

    def check_data_type(self, df: pd.DataFrame):
        pass

    def check_discount(self, df: pd.DataFrame, col_nm: str):
        pass

    def make_err_format(self, df: pd.DataFrame, err_cd: str):
        df['project_cd'] = self.common['project_cd']
        df['data_vrsn_cd'] = self.data_vrsn_cd
        df['err_grp_cd'] = self.err_grp_map[err_cd]
        df['err_cd'] = err_cd.upper()
        if self.division == 'sell_out':
            df['from_dc_cd'] = ''
        df['create_user_cd'] = 'SYSTEM'

        df = df.rename(columns={'biz_cd': 'item_attr01_cd',
                                'line_cd': 'item_attr02_cd',
                                'brand_cd': 'item_attr03_cd',
                                'item_cd': 'item_attr04_cd',
                                'sku_cd': 'item_attr05_cd'})

        return df
