import common.util as util

import pandas as pd


class DataPrep(object):
    DROP_COL_SALES = ['division_cd', 'seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'create_date',
                      'yymmdd', 'week', 'cust_cd']
    STR_TYPE_COLS = ['cust_cd', 'sku_cd']

    def __init__(self, division: str, common: dict, exg_list: list):
        self.division = division
        self.input_cols = ['discount'] + exg_list
        self.target_col = common['target_col']

    def preprocess(self, sales: pd.DataFrame, exg: dict):
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # convert data type
        for col in self.STR_TYPE_COLS:
            sales[col] = sales[col].astype(str)

        # Drop columns
        sales = sales.drop(columns=self.DROP_COL_SALES)

        # ------------------------------- #
        # 2. Preprocess Exogenous dataset
        # ------------------------------- #
        exg_all = util.prep_exg_all(data=exg['all'])
        exg_partial = util.prep_exg_partial(data=exg['partial'])

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Merge sales data & exogenous data
        data = pd.merge(sales, exg_all, on='yymmdd', how='left')

        print("")
