import common.util as util
import common.config as config
from dao.DataIO import DataIO
from operation.Cycle import Cycle
from common.SqlConfig import SqlConfig

from typing import List, Tuple, Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class SalesAnalysis(object):
    STR_TYPE_COLS = ['cust_grp_cd', 'sku_cd']

    def __init__(self, step_cfg: dict, data_cfg: dict):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.path_root = os.path.join('..', '..')
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.hist_date_list = None
        self.hist_date_cnt = 0

        # Data configuration
        self.step_cfg = step_cfg
        self.data_cfg = data_cfg
        self.division = data_cfg['division']
        self.cut_lvl = 10

        # information
        self.date = {}
        self.path = {}
        self.level = {}
        self.hrchy = {}
        self.hrchy_level = 0
        self.data_vrsn_cd = ''

    def run(self):
        self.init()

        sales = None
        if self.step_cfg['cls_load']:
            sales = self.load_sales()
            # compare = self.load_compare()

        sales_raw, accuracy = [], []
        if self.step_cfg['cls_prep']:
            if not self.step_cfg['cls_load']:
                sales = self.io.load_object(file_path=self.path['load'], data_type='csv')
                accuracy = self.load_accuracy()
                sales_raw, sales_grp = self.preprocess(data=sales)

        if self.step_cfg['cls_comp']:
            self.compare_result(sales=sales_raw, accuracy=accuracy)

        if self.step_cfg['cls_view']:
            self.view(data=sales_raw)

        print("Finished")

    def compare_result(self, sales: pd.DataFrame, accuracy: pd.DataFrame):

        accuracy['accuracy'] = np.where(accuracy['accuracy'] > 1, 2 - accuracy['accuracy'], accuracy['accuracy'])
        accuracy['accuracy'] = np.where(accuracy['accuracy'] < 0, 0, accuracy['accuracy'])

        merged = pd.merge(
            accuracy,
            sales,
            on=['cust_grp_cd', 'item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd'],
            how='inner'
        )

        # Filter zero sales
        if self.data_cfg['rm_zero_yn']:
            merged = merged[merged['sales'] != 0]

        merged = merged.rename(columns={'bins': 'bin_cnt'})
        merged['bin_acc'] = pd.cut(
            merged['accuracy'],
            bins=[num / 10 for num in range(0, 12, 1)],
            right=False)

        merged_grp = merged.groupby(by=['bin_acc', 'bin_cnt'])['success'].count().reset_index()
        merged_grp_pivot = merged_grp.pivot(index='bin_cnt', columns='bin_acc', values='success')
        merged_grp_pivot = merged_grp_pivot.rename(columns={'success': 'cnt'})

        if self.data_cfg['rm_zero_yn']:
            folder_nm = 'without_zero'
        else:
            folder_nm = 'with_zero'

        merged_grp_pivot.to_csv(
            os.path.join(self.path['cnt_acc'], folder_nm, self.data_vrsn_cd + '_' + self.division + '_' +
                         str(self.level['item_lvl']) + '.csv'))

    def init(self):
        self.set_date()
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()
        self.set_path()

    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Drop unnecessary columns
        drop_col = ['seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'discount', 'create_date']
        data = data.drop(columns=drop_col, errors='ignore')

        # convert data type
        for col in self.STR_TYPE_COLS:
            data[col] = data[col].astype(str)

        data = data.dropna(axis=0, subset=['biz_cd'])

        data = self.conv_data_type(df=data)

        data_group, hrchy_cnt = util.group(
            hrchy=self.hrchy['apply'],
            hrchy_lvl=self.hrchy_level,
            data=data
        )

        # Resampling
        data_resample = util.hrchy_recursion_with_none(
            hrchy_lvl=self.hrchy_level,
            fn=self.resample,
            df=data_group
        )

        result_raw = util.hrchy_recursion_extend_key(
            hrchy_lvl=self.hrchy_level,
            fn=self.conv_to_df,
            df=data_resample
        )
        # Choose count column
        # result_cnt = np.array(result_raw)[:, -1].astype(int).tolist()

        df_raw, df_grp = self.count_by_level(data=result_raw)
        df_grp.to_csv(
            os.path.join(
            self.path['rate'], self.data_vrsn_cd + '_' + self.division + '_' + str(self.level['item_lvl']) + '.csv'),
            index=False
        )

        # Rename columns
        df_raw.columns = [config.HRCHY_CD_TO_DB_CD_MAP.get(col, col) for col in df_raw.columns]
        df_grp.columns = [config.HRCHY_CD_TO_DB_CD_MAP.get(col, col) for col in df_grp.columns]

        return df_raw, df_grp

    def view(self, data: list) -> None:
        plt.hist(x=data, bins=30)
        plt.savefig(os.path.join(
            self.path['view'], self.data_vrsn_cd + '_' + self.division + '_' + str(self.level['item_lvl']) + '.png'))

    def count_by_level(self, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = self.hrchy['apply'] + ['cnt']
        df_raw = pd.DataFrame(data, columns=cols)
        df_raw['bins'] = pd.cut(df_raw['cnt'], bins=range(0, 170, 10), right=False)

        # Grouping
        df_grp = df_raw.groupby('bins').count().reset_index()
        df_grp = df_grp.sort_values(by='bins', ascending=False)
        df_grp['cum_cnt'] = df_grp['cnt'].cumsum(axis=0)
        total = df_grp['cnt'].sum()
        df_grp['rev_cnt'] = total - df_grp['cum_cnt']
        df_grp['rev_pct'] = df_grp['rev_cnt'] / total

        return df_raw, df_grp

    def set_date(self) -> None:
        if self.data_cfg['cycle_yn']:
            cycle = Cycle(common=self.common, rule='w')
            cycle.calc_period()
            date = cycle.hist_period
            self.date = {'from': date[0], 'to': date[1]}
            self.data_vrsn_cd = self.date['from'] + '-' + self.date['to']
        else:
            self.date = self.data_cfg['date']
            self.data_vrsn_cd = self.date['from'] + '-' + self.date['to']

        # set history range
        self.hist_date_list = pd.date_range(
            start=self.date['from'],
            end=self.date['to'],
            freq='W-MON'
        )
        self.hist_date_cnt = len(self.hist_date_list)

    def set_level(self,  item_lvl: int) -> None:
        level = {
            'cust_lvl': 1,    # Fixed
            'item_lvl': item_lvl,
        }
        self.level = level

    def set_hrchy(self) -> None:
        self.hrchy = {
            'cnt': 0,
            'key': "C" + str(self.level['cust_lvl']) + '-' + "P" + str(self.level['item_lvl']) + '-',
            'lvl': {
                'cust': self.level['cust_lvl'],
                'item': self.level['item_lvl'],
                'total': self.level['cust_lvl'] + self.level['item_lvl']
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),
                'item': self.common['hrchy_item'].split(',')[:self.level['item_lvl']]
            }
        }
        self.hrchy['apply'] = self.hrchy['list']['cust'] + self.hrchy['list']['item']
        self.hrchy_level = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1

    def set_path(self) -> None:
        self.path = {
            'load': util.make_path_baseline(
                path=self.path_root, module='data', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl='', step='load', extension='csv'),
            'rate': os.path.join(self.path_root, 'analysis', 'sales', 'rate'),
            'cnt_acc': os.path.join(self.path_root, 'analysis', 'sales', 'cnt_acc'),
            'view': os.path.join(self.path_root, 'analysis', 'sales', 'hist')
        }

    def load_sales(self) -> pd.DataFrame:
        sales = None
        if self.division == 'SELL_IN':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))

        elif self.division == 'SELL_OUT':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**self.date))

        return sales

    def load_accuracy(self):
        path = os.path.join(self.path_root, 'analysis', 'accuracy', self.data_vrsn_cd + '_' +
                            self.division + '_' + str(self.hrchy['lvl']['item']) + '.csv')
        compare = self.io.load_object(file_path=path, data_type='csv')

        compare['cust_grp_cd'] = compare['cust_grp_cd'].astype(str)

        return compare

    @staticmethod
    def conv_to_df(hrchy: list, data) -> List[List[Union[str, int]]]:
        return [hrchy + [data]]

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.common['date_col']] = pd.to_datetime(df[self.common['date_col']], format='%Y%m%d')
        df = df.set_index(keys=[self.common['date_col']])

        return df

    def resample(self, df: pd.DataFrame) -> int:
        resampled = df.resample(rule='W-MON').sum()

        if len(resampled.index) < self.hist_date_cnt:
            # missed_rate = self.check_missing_data(df=df_resampled)
            resampled = self.fill_missing_date(df=resampled)

        cnt = self.chk_backward_zero(data=resampled)

        return cnt

    def chk_backward_zero(self, data: pd.DataFrame) -> int:
        trimmed = len(np.trim_zeros(data.to_numpy(), trim='b'))
        cnt = self.hist_date_cnt - trimmed
        if cnt < 0:
            cnt = 0

        return cnt

    def fill_missing_date(self, df: pd.DataFrame) -> pd.DataFrame:
        idx_add = list(set(self.hist_date_list) - set(df.index))
        data_add = np.zeros((len(idx_add), df.shape[1]))
        df_add = pd.DataFrame(data_add, index=idx_add, columns=df.columns)
        df = df.append(df_add)
        df = df.sort_index()

        return df
