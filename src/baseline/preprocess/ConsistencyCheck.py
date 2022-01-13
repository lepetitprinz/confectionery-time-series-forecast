import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import pandas as pd
from copy import deepcopy


class ConsistencyCheck(object):
    def __init__(self, data_vrsn_cd: str, division: str, common: dict, hrchy: dict,
                 mst_info: dict, exec_cfg: dict, err_grp_map: dict, path_root: str):
        # Class Configuration
        self.io = DataIO()
        self.sql_config = SqlConfig()

        # Data Configuration
        self.exec_cfg = exec_cfg
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division
        self.common = common
        self.hrchy = hrchy

        self.resample_sum_list = ['qty']
        self.resample_avg_list = ['unit_price', 'discount']

        # Information
        self.mst_info = mst_info
        self.err_grp_map = err_grp_map
        self.unit_cd = common['unit_cd'].split(',')
        self.col_numeric = ['unit_price', 'discount', 'qty']
        self.resample_rule = 'W-MON'

        # Save and Load Configuration
        self.save_path = os.path.join(path_root, 'error')
        self.tb_name = 'M4S_I002174'

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        # Code Mapping
        normal = self.check_code_map(df=df)

        # Filter sales matrix
        # normal = self.merge_sales_matrix(df=normal)

        return normal

    def check_code_map(self, df: pd.DataFrame) -> pd.DataFrame:
        normal = self.check_prod_level(df=df)

        return normal

    # Error 1
    def check_prod_level(self, df: pd.DataFrame):
        test_df = deepcopy(df)
        test_df = test_df[self.hrchy['apply']]
        na_rows = test_df.isna().sum(axis=1) > 0
        err = df[na_rows].fillna('')
        normal = df[~na_rows]

        # Print result
        print("---------------------------------------")
        print(f"Normal data length: {len(normal)}")
        print(f"Error data length: {len(err)}")
        print("---------------------------------------")

        # save the error data
        err_cd = 'err001'
        err = self.make_err_format(df=err, err_cd=err_cd)

        # Save results
        if self.exec_cfg['save_step_yn']:
            path = os.path.join(self.save_path, self.division + '-' + self.data_vrsn_cd + '-' + 'cns' +
                                '-' + err_cd + '.csv')
            err.to_csv(path, index=False, encoding='cp949')

        if len(err) > 0 and self.exec_cfg['save_db_yn']:
            info = {'data_vrsn_cd': self.data_vrsn_cd, 'division_cd': self.division, 'err_cd': err_cd}
            self.io.delete_from_db(sql=self.sql_config.del_sales_err(**info))
            self.io.insert_to_db(df=err, tb_name=self.tb_name)

        return normal

    def merge_sales_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sku_cd'] = df['sku_cd'].astype(str)

        merged = pd.merge(
            df,
            self.mst_info['sales_matrix'],
            on=['cust_grp_cd', 'sku_cd'],
            how='inner',
            suffixes=('', '_DROP')
        ).filter(regex='^(?!.*_DROP)')

        print(f"Sale matrix filtered data length: {len(merged)}\n")
        print("---------------------------------------")

        return merged

    def fill_numeric_na(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.col_numeric:
            data[col] = data[col].fillna(0)    # fill na to zero
            data[col] = data[col].replace('', 0)    # replace empty string to zero
        return data

    def resample_by_agg(self, df: pd.DataFrame, agg: str) -> pd.DataFrame:
        resampled = None
        if agg == 'sum':
            resampled = df.resample(rule=self.resample_rule).sum()
        elif agg == 'avg':
            resampled = df.resample(rule=self.resample_rule).mean()
        resampled = resampled.fillna(value=0)    # fill NaN
        resampled = resampled.reset_index()

        return resampled

    def group_by_week(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert date string to datetime
        df['yymmdd'] = pd.to_datetime(df['yymmdd'], format='%Y%m%d')
        df = df.set_index(keys=['yymmdd'])

        # convert date type
        df['cust_grp_cd'] = df['cust_grp_cd'].astype(str)
        df['sku_cd'] = df['sku_cd'].astype(str)

        # Get unique group
        grp_col = ['cust_grp_cd', 'sku_cd']
        grp_df = df[grp_col].drop_duplicates()
        grp_list = [tuple(x) for x in grp_df.to_numpy()]

        result = pd.DataFrame()
        for cust_grp, sku in grp_list:
            temp = df[(df['cust_grp_cd'] == cust_grp) & (df['sku_cd'] == sku)]
            resampled_sum = self.resample_by_agg(df=temp[grp_col + self.resample_sum_list], agg='sum')
            resampled_avg = self.resample_by_agg(df=temp[grp_col + self.resample_avg_list], agg='avg')
            resampled = pd.merge(resampled_sum, resampled_avg, on='yymmdd')

            # add information
            resampled['cust_grp_cd'] = cust_grp
            resampled['sku_cd'] = sku

            result = pd.concat([result, resampled], axis=0)

        result = result.reset_index(drop=True)

        return result

    def make_err_format(self, df: pd.DataFrame, err_cd: str):
        df = self.group_by_week(df=df)

        df['project_cd'] = self.common['project_cd']
        df['data_vrsn_cd'] = self.data_vrsn_cd
        df['division_cd'] = self.division
        df['err_grp_cd'] = self.err_grp_map[err_cd]
        df['err_cd'] = err_cd.upper()
        df['from_dc_cd'] = ''
        df['create_user_cd'] = 'SYSTEM'

        # Merge SP1 information
        cust_grp = self.mst_info['cust_grp']
        cust_grp['cust_grp_cd'] = cust_grp['cust_grp_cd'].astype(str)
        df['cust_grp_cd'] = df['cust_grp_cd'].astype(str)
        df = pd.merge(df, cust_grp, on='cust_grp_cd', how='left')

        # Merge item master information
        item_mst = self.mst_info['item_mst']
        item_mst['sku_cd'] = item_mst['sku_cd'].astype(str)
        df['sku_cd'] = df['sku_cd'].astype(str)
        df = pd.merge(df, item_mst, on='sku_cd', how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # Fill na
        df = self.fill_numeric_na(data=df)
        df = df.fillna('')

        # Rename columns
        df = df.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)
        df = df.rename(columns=config.HRCHY_SKU_TO_DB_SKU_MAP)

        number = [str(i+1).zfill(10) for i in range(len(df))]
        df['number'] = number
        df['seq'] = df['seq'] + '-' + df['number']

        df = df.drop(columns=['number', 'create_date'], errors='ignore')

        return df

    # Error 2
    # def check_unit_code(self, df: pd.DataFrame):
    #     err = df[~df['unit_cd'].isin(self.unit_cd)]
    #     normal = df[df['unit_cd'].isin(self.unit_cd)]
    #
    #     err_cd = 'err002'
    #     err = self.make_err_format(df=err, err_cd=err_cd)
    #
    #     if self.exec_cfg['save_step_yn']:
    #         path = os.path.join(
    #             self.save_path, self.division + '-' + self.data_vrsn_cd + '-' + 'cns' + '-' +
    #             err_cd + '.csv')
    #         err.to_csv(path, index=False, encoding='cp949')
    #
    #     if len(err) > 0 and self.exec_cfg['save_db_yn']:
    #         info = {'data_vrsn_cd': self.data_vrsn_cd, 'division_cd': self.division, 'err_cd': err_cd}
    #         self.io.delete_from_db(sql=self.sql_config.del_sales_err(**info))
    #         self.io.insert_to_db(df=err, tb_name=self.tb_name)
    #
    #     return normal

    # Error 3
    # def check_unit_code_map(self, df: pd.DataFrame):
    #     unit_code_map = self.io.get_df_from_db(sql=self.sql_config.sql_unit_map())
    #     unit_code_map.columns = [col.lower() for col in unit_code_map.columns]
    #     df['sku_cd'] = df['sku_cd'].astype(str)
    #
    #     # Merge unit code map
    #     merged = pd.merge(df, unit_code_map, how='left', on='sku_cd')
    #     err = merged[merged['box_bol'].isna()]
    #     normal = merged[~merged['box_bol'].isna()]
    #
    #     # Save Error
    #     err_cd = 'err003'
    #     err = self.make_err_format(df=err, err_cd=err_cd)
    #
    #     # Save result
    #     if self.exec_cfg['save_step_yn']:
    #         path = os.path.join(
    #             self.save_path, self.division + '-' + self.data_vrsn_cd + '-' + 'cns' + '-' +
    #             err_cd + '.csv')
    #         err.to_csv(path, index=False, encoding='cp949')
    #
    #     if len(err) > 0 and self.exec_cfg['save_db_yn']:
    #         info = {'data_vrsn_cd': self.data_vrsn_cd, 'division_cd': self.division, 'err_cd': err_cd}
    #         self.io.delete_from_db(sql=self.sql_config.del_sales_err(**info))
    #         self.io.insert_to_db(df=err, tb_name=self.tb_name)
    #
    #     return normal
