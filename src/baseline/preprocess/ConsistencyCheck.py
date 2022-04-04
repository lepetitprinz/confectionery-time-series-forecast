import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import pandas as pd
import datetime
from copy import deepcopy


class ConsistencyCheck(object):
    item_class_list = ['10', '20', '60']    # Applying item class list

    def __init__(self, data_vrsn_cd: str, division: str, common: dict, hrchy: dict,
                 mst_info: dict, exec_cfg: dict, err_grp_map: dict, path_root: str):
        """
        :param data_vrsn_cd: Data version
        :param division: Division (SELL-IN/SELL-OUT)
        :param common: Common information
        :param hrchy: Hierarchy information
        :param mst_info: Master information
        :param exec_cfg: Execution configuration
        :param err_grp_map: Error group information
        :param path_root: Root path for baseline forecast
        """

        # Class Configuration
        self.io = DataIO()
        self.sql_config = SqlConfig()

        # Data instance attribute
        self.exec_cfg = exec_cfg
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division
        self.common = common
        self.hrchy = hrchy

        # Resampling instance attribute
        self.resample_rule = 'W-MON'    # Resampling rule
        self.resample_sum_list = ['qty']     # columns of weekly resampling rule (Summation)
        self.resample_avg_list = ['unit_price', 'discount']    # columns of weekly resampling rule (Average)

        # Information instance attribute
        self.mst_info = mst_info
        self.err_grp_map = err_grp_map
        self.unit_cd = common['unit_cd'].split(',')    # Applying unit code list
        self.col_numeric = ['unit_price', 'discount', 'qty']    # Numeric column list

        # Save and Load instance attribute
        self.save_path = os.path.join(path_root, 'error')    # Save path
        self.tb_name = 'M4S_I002174'    # Saving table name

    # Check the data consistency
    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        # Code Mapping
        normal = self.check_code_map(df=df)

        # Filter sales matrix
        # normal = self.merge_sales_matrix(df=normal)

        return normal

    def check_code_map(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check item Mapping
        normal = self.check_item_map(df=df)

        # Check customer <-> SP1 Mapping
        self.check_cust_map()

        return normal

    def check_cust_map(self) -> None:
        today = datetime.date.today()    # Today date (YYYY-MM-DD)
        today = today - datetime.timedelta(days=today.weekday())    # This monday

        date_from = today - datetime.timedelta(days=7)    # Previous monday
        date_to = today - datetime.timedelta(days=1)      # Previous sunday

        # Convert datetime to string type
        date_from = date_from.strftime('%Y%m%d')
        date_to = date_to.strftime('%Y%m%d')
        date = {'from': date_from, 'to': date_to}

        # Get customer
        result = self.io.get_df_from_db(sql=self.sql_config.sql_cust_sp1_map_error(**date))

        # Filter customer distribution
        # result = self.filter_customer_distribution(df=result)

        # Save the result
        save_path = os.path.join(self.save_path, 'sp1', self.division + '_' + date_from + '_' + date_to + '.csv')
        self.io.save_object(data=result, data_type='csv', file_path=save_path)

    # filter customer distribution
    def filter_customer_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        df['item_class_cd'] = df['item_class_cd'].astype(str)
        df = df[df['item_class_cd'].isin(self.item_class_list)]

        return df

    # Error 1
    # Check the item information mapping error
    def check_item_map(self, df: pd.DataFrame) -> pd.DataFrame:
        test_df = deepcopy(df)
        test_df = test_df[self.hrchy['apply']]
        na_rows = test_df.isna().sum(axis=1) > 0
        err = df[na_rows].fillna('')
        normal = df[~na_rows]

        # Print result
        print("-----------------------------------")
        print(f"All data length: {len(test_df)}")
        print(f"Normal data length: {len(normal)}")
        print(f"Error data length: {len(err)}")
        print("-----------------------------------")

        # Save the error data
        err_cd = 'err001'
        if len(err) > 0:
            err = self.make_err_format(df=err, err_cd=err_cd)

            # Save results
            if self.exec_cfg['save_step_yn']:
                path = os.path.join(self.save_path, self.division + '-' + self.data_vrsn_cd + '-' + 'cns' +
                                    '-' + err_cd + '.csv')
                err.to_csv(path, index=False, encoding='cp949')

            if self.exec_cfg['save_db_yn']:
                info = {'data_vrsn_cd': self.data_vrsn_cd, 'division_cd': self.division, 'err_cd': err_cd}
                self.io.delete_from_db(sql=self.sql_config.del_sales_err(**info))    # Delete duplicated result
                self.io.insert_to_db(df=err, tb_name=self.tb_name)    # Insert result on DB

        return normal

    # Demand forecast based on sp1 + items contained in sales matrix
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

    # Fill empty rows with zeros
    def fill_numeric_na(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.col_numeric:
            data[col] = data[col].fillna(0)    # Fill empty rows with zeros
            data[col] = data[col].replace('', 0)    # replace empty string to zero
        return data

    # Resampling
    def resample_by_agg(self, df: pd.DataFrame, agg: str) -> pd.DataFrame:
        resampled = None
        if agg == 'sum':    # Summation
            resampled = df.resample(rule=self.resample_rule).sum()
        elif agg == 'avg':    # Average
            resampled = df.resample(rule=self.resample_rule).mean()
        resampled = resampled.fillna(value=0)    # Fill empty rows with
        resampled = resampled.reset_index()      # Reset index

        return resampled

    def group_by_week(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert date string into datetime
        df['yymmdd'] = pd.to_datetime(df['yymmdd'], format='%Y%m%d')
        df = df.set_index(keys=['yymmdd'])

        # Change date type
        df['cust_grp_cd'] = df['cust_grp_cd'].astype(str)
        df['sku_cd'] = df['sku_cd'].astype(str)

        # Make unique group (SP1 + Customer)
        grp_col = ['cust_grp_cd', 'sku_cd']
        grp_df = df[grp_col].drop_duplicates()
        grp_list = [tuple(x) for x in grp_df.to_numpy()]

        result = pd.DataFrame()
        for cust_grp, sku in grp_list:
            temp = df[(df['cust_grp_cd'] == cust_grp) & (df['sku_cd'] == sku)]
            resampled_sum = self.resample_by_agg(df=temp[grp_col + self.resample_sum_list], agg='sum')
            resampled_avg = self.resample_by_agg(df=temp[grp_col + self.resample_avg_list], agg='avg')
            resampled = pd.merge(resampled_sum, resampled_avg, on='yymmdd')

            # Add information
            resampled['cust_grp_cd'] = cust_grp
            resampled['sku_cd'] = sku

            result = pd.concat([result, resampled], axis=0)

        result = result.reset_index(drop=True)

        return result

    # Convert error result into db format
    def make_err_format(self, df: pd.DataFrame, err_cd: str) -> pd.DataFrame:
        df = self.group_by_week(df=df)

        # Add columns
        df['project_cd'] = self.common['project_cd']    # Project code
        df['data_vrsn_cd'] = self.data_vrsn_cd          # Data version
        df['division_cd'] = self.division               # Division(SELL-IN/SELL-OUT)
        df['err_grp_cd'] = self.err_grp_map[err_cd]     # Error group information
        df['err_cd'] = err_cd.upper()                   # Error code
        df['from_dc_cd'] = ''                           # From DC code
        df['create_user_cd'] = 'SYSTEM'                 # Create user code

        # Change data types (Integer -> String)
        cust_grp = self.mst_info['cust_grp']
        cust_grp['cust_grp_cd'] = cust_grp['cust_grp_cd'].astype(str)
        df['cust_grp_cd'] = df['cust_grp_cd'].astype(str)

        # Merge SP1 information
        df = pd.merge(df, cust_grp, on='cust_grp_cd', how='left')

        # Merge item master information
        item_mst = self.mst_info['item_mst']
        item_mst['sku_cd'] = item_mst['sku_cd'].astype(str)
        df['sku_cd'] = df['sku_cd'].astype(str)
        df = pd.merge(df, item_mst, on='sku_cd', how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # Fill empty rows with zeros and empty space
        df = self.fill_numeric_na(data=df)
        df = df.fillna('')

        # Rename columns
        df = df.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)
        df = df.rename(columns=config.HRCHY_SKU_TO_DB_SKU_MAP)

        number = [str(i+1).zfill(10) for i in range(len(df))]    # Numbering sequence
        df['number'] = number
        df['seq'] = df['seq'] + '-' + df['number']    # Add sequnce column

        # Remove unnecessary columns
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
