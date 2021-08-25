from dao.SqlSession import SqlSession
from dao.SqlConfig import SqlConfig

from copy import deepcopy
import pandas as pd


class ConsistencyCheck(object):
    PROD_LVL_LIST = ['biz_cd', 'line_cd', 'brand_cd', 'item_ctgr_cd']
    UNIT_CD = ['BOX', 'EA ', 'BOL']
    ERR_CD_MAP = {'ERR001': 'Item - Item Level mapping 데이터 누락',
                  'ERR002': '존재하지 않는 단위',
                  'ERR003': 'BOX로 환산 단위(EA, BOL) mapping 데이터 누락'
                  }

    def __init__(self, division: str, save_yn: bool):
        self.cns_tb_name = 'M4S_O000000'
        self.division = division
        self.save_yn = save_yn
        self.err_table_cols = ['project_cd', 'division_cd', 'data_vrsn_cd', 'err_cd', 'err_nm',
                               'sold_cust_grp_cd', 'item_cd', 'biz_cd', 'line_cd', 'brand_cd', 'item_ctgr_cd',
                               'yymmdd', 'seq', 'from_dc_cd', 'unit_price', 'unit_cd', 'discount', 'week',
                               'qty', 'create_user_cd', 'create_date', 'modify_user_cd', 'modify_date']
        self.sql_config = SqlConfig()
        self.session = SqlSession()
        self.session.init()

    def check(self, df: pd.DataFrame):
        # convert to lowercase columns
        df.columns = [col.lower() for col in df.columns]

        # Code Mapping
        normal = self.check_code_map(df=df)

        return normal

    def check_code_map(self, df: pd.DataFrame):
        normal = self.check_prod_level(df=df)
        normal = self.check_unit_code(df=normal)
        normal = self.check_unit_code_map(df=normal)

        return normal

    # Error 1
    def check_prod_level(self, df: pd.DataFrame):
        test_df = deepcopy(df)
        test_df = test_df[self.PROD_LVL_LIST]
        na_rows = test_df.isna().sum(axis=1) > 0
        err = df[na_rows].fillna('')
        normal = df[~na_rows]

        # save the error data
        err = self.make_err_format(df=err, err_cd='ERR001')
        if len(err) > 0 and self.save_yn:
            self.session.update(df=err, tb_name=self.cns_tb_name)

        return normal

    # Error 2
    def check_unit_code(self, df: pd.DataFrame):
        err = df[~df['unit_cd'].isin(self.UNIT_CD)]
        normal = df[df['unit_cd'].isin(self.UNIT_CD)]

        err = self.make_err_format(df=err, err_cd='ERR002')
        if len(err) > 0 and self.save_yn:
            self.session.update(df=err, tb_name=self.cns_tb_name)

        return normal

    # Error 3
    def check_unit_code_map(self, df: pd.DataFrame):
        unit_code_map = self.session.select(sql=self.sql_config.sql_unit_map())
        unit_code_map.columns = [col.lower() for col in unit_code_map.columns]
        df['item_cd'] = df['item_cd'].astype(str)
        merged = pd.merge(df, unit_code_map, how='left', on='item_cd')
        err = merged[merged['box_bol'].isna()]
        normal = merged[~merged['box_bol'].isna()]

        # Save Error
        err = self.make_err_format(df=err, err_cd='ERR003')
        if len(err) > 0 and self.save_yn:
            self.session.update(df=err, tb_name=self.cns_tb_name)

        return normal

    def check_data_type(self, df: pd.DataFrame):
        pass

    def check_discount(self, df: pd.DataFrame, col_nm: str):
        pass

    @staticmethod
    def fill_na(df: pd.DataFrame):
        is_null_col = [col for col, is_null in zip(df.columns, df.isnull().sum()) if is_null > 0]
        for col in is_null_col:
            df[col] = df[col].fillna(0)

    def make_err_format(self, df: pd.DataFrame, err_cd: str):
        df['project_cd'] = 'ENT001'
        df['data_vrsn_cd'] = '-'    # temp
        df['err_cd'] = err_cd
        df['err_nm'] = self.ERR_CD_MAP[err_cd]
        if self.division == 'sell_out':
            df['from_dc_cd'] = ''
        df['create_user_cd'] = 'SYSTEM'
        df['modify_user_cd'] = ''
        df['modify_date'] = ''

        df = df[self.err_table_cols]
        df.columns = [col.upper() for col in df.columns]

        return df

    def save_result(self):
        pass