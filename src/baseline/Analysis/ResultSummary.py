import common.util as util

import numpy as np
import pandas as pd
from copy import deepcopy


class ResultSummary(object):
    drop_col1 = ['project_cd', 'data_vrsn_cd', 'fkey', 'create_user_cd']
    hrchy_name_map = {'biz_cd': 'biz_nm', 'line_cd': 'line_nm', 'brand_cd': 'brand_nm', 'item_cd': 'item_nm'}
    item_name_map = {
        'biz_cd': 'item_attr01_cd', 'line_cd': 'item_attr02_cd', 'brand_cd': 'item_attr03_cd',
        'item_cd': 'item_attr04_cd', 'biz_nm': 'item_attr01_nm', 'line_nm': 'item_attr02_nm',
        'brand_nm': 'item_attr03_nm', 'item_nm': 'item_attr04_nm'
    }
    sku_name_map = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}

    def __init__(self, data_vrsn: str, division: str, date: dict, common: dict,  test_vrsn: str,
                 hrchy: dict, item_mst: pd.DataFrame, lvl_cfg: dict):
        # Data Information Configuration
        self.data_vrsn = data_vrsn
        self.division = division
        self.date = date
        self.common = common
        self.item_mst = item_mst
        self.test_vrsn = test_vrsn
        self.lvl_cfg = lvl_cfg

        # Data Level Configuration
        self.hrchy = hrchy
        self.hrchy_item_cd_list = common['db_hrchy_item_cd'].split(',')
        self.hrchy_item_nm_list = common['db_hrchy_item_nm'].split(',')

        self.str_type_cols = ['cust_grp_cd', 'item_cd', self.common['date_col']]
        self.key_col = ['cust_grp_cd', 'item_cd']
        self.sku_col = 'item_cd'
        self.pred_date_range = pd.date_range(
            start=common['pred_start_day'],
            end=common['pred_end_day'],
            freq=common['resample_rule']
        )
        self.save_path = {
            'all': util.make_path_baseline(
                path=self.common['path_local'], module='report', division=division, data_vrsn=data_vrsn,
                hrchy_lvl=hrchy['key'], step='all', extension='csv'),
            'summary': util.make_path_baseline(
                path=self.common['path_local'], module='report', division=division, data_vrsn=data_vrsn,
                hrchy_lvl=hrchy['key'], step='summary', extension='csv')
        }

    def compare_result(self, sales_comp, sales_recent, pred):
        raw_all = self.make_raw_result(
            sales_comp=sales_comp,
            sales_recent=sales_recent,
            pred=pred)
        # self.make_summary(df=raw_all)

        return raw_all

    def make_raw_result(self, sales_comp, sales_recent, pred):
        # convert lower
        sales_comp = util.conv_col_lower(data=sales_comp)
        sales_recent = util.conv_col_lower(data=sales_recent)
        pred = util.conv_col_lower(data=pred)

        # rename columns
        sales_comp = self.rename_cols(data=sales_comp)
        sales_recent = self.rename_cols(data=sales_recent)
        pred = self.rename_cols(data=pred)
        self.item_mst = self.rename_cols(data=self.item_mst)

        pred['division_cd'] = self.division
        pred = self.conv_data_type(data=pred)

        hrchy_item = None
        if (not self.lvl_cfg['middle_out']) and (self.hrchy['lvl']['item'] != 5):
            sales_recent, _ = self.resample_sales(data=sales_recent)
            sales_comp, hrchy_item = self.resample_sales(data=sales_comp)

        # If execute middle out
        elif self.lvl_cfg['middle_out']:
            pred = pd.merge(
                pred, self.item_mst, how='left', on=self.hrchy_item_cd_list[:-1] + [self.sku_col],
                suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        pred, name_rev_map = self.filter_recent_exist_sales(sales_recent=sales_recent, pred=pred)
        pred = self.prep_prediction(data=pred)

        item_col = ''
        if self.lvl_cfg['middle_out']:
            item_col = ['item_cd']
        else:
            item_col = [name_rev_map[self.hrchy['apply'][-1]]]

        merge_col = ['division_cd', 'yy', 'week', 'cust_grp_cd'] + item_col
        merged = pd.merge(
            pred,
            sales_comp,
            on=merge_col,
            how='left',
            suffixes=('', '_DROP')
        ).filter(regex='^(?!.*_DROP)')

        # Fill NA weeks with 0 qty
        merged = self.fill_na_week(data=merged)

        # Drop dates that doesn't compare
        merged = merged.fillna(0)

        # Calculate absolute difference
        merged['diff'] = merged['sales'] - merged['pred']
        merged['diff'] = np.round(np.absolute(merged['diff'].to_numpy()), 2)

        # merged = merged.rename(columns=self.item_name_rev_map)
        # merged = merged.rename(columns={'sku_cd': 'item_cd', 'sku_nm': 'item_nm'})

        # Sort columns
        fixed_col = ['cust_grp_cd', 'cust_grp_nm', 'stat_cd', 'yy', 'week', 'sales', 'pred', 'diff']
        if (self.lvl_cfg['middle_out']) or (self.hrchy['lvl']['item'] == 5):
            hrchy_item = self.hrchy_item_cd_list + self.hrchy_item_nm_list

        merged = merged[hrchy_item + fixed_col]

        return merged

    def rename_cols(self, data: pd.DataFrame):
        if not 'item_attr01_cd' in data.columns:
            data = data.rename(columns=self.item_name_map)
        if 'sku_cd' in data.columns:
            data = data.rename(columns=self.sku_name_map)

        return data

    def make_db_format(self, data):
        data['seq'] = [self.test_vrsn + '_' + str(i+1).zfill(7) for i in range(len(data))]
        data['project_cd'] = self.common['project_cd']
        data['data_vrsn_cd'] = self.data_vrsn
        data['division_cd'] = self.division
        data['test_vrsn_cd'] = self.test_vrsn
        if (self.lvl_cfg['middle_out']) or (self.hrchy['lvl']['item'] == 5):
            data['fkey'] = 'C1-P5'
        else:
            data['fkey'] = self.hrchy['key'][:-1]
        data['create_user_cd'] = 'SYSTEM'

        info = self.make_del_info()

        return data, info

    def make_del_info(self):
        info = {
            'project_cd': self.common['project_cd'],
            'data_vrsn_cd': self.data_vrsn,
            'division_cd': self.division,
            'test_vrsn_cd': self.test_vrsn
        }

        return info

    def filter_recent_exist_sales(self, sales_recent: pd.DataFrame, pred: pd.DataFrame):
        # Rename sales columns
        name_rev_map = deepcopy(self.item_name_map)
        name_rev_map.update(self.sku_name_map)

        if self.lvl_cfg['middle_out']:
            key_col = ['cust_grp_cd', 'item_cd']
        else:
            key_col = ['cust_grp_cd', name_rev_map[self.hrchy['apply'][-1]]]

        key_col_df = sales_recent[key_col].drop_duplicates().reset_index(drop=True)
        key_col_df['key'] = key_col_df[key_col[0]] + '-' + key_col_df[key_col[1]]
        pred['key'] = pred[key_col[0]] + '-' + pred[key_col[1]]
        masked = pred[pred['key'].isin(key_col_df['key'])]
        masked = masked.drop(columns=['key'])

        return masked, name_rev_map

    def prep_prediction(self, data):
        # Preprocess the prediction dataset
        # Rename columns
        data = data.rename(columns={'sales': 'pred', 'result_sales': 'pred'})
        data = data.drop(columns=self.drop_col1, errors='ignore')

        # Round prediction results
        data['pred'] = np.round(data['pred'].to_numpy(), 2)
        data['yy'] = data[self.common['date_col']].str.slice(0, 4)

        # Filter comparing days
        data = data[data['yymmdd'] >= self.date['evaluation']['from']]
        data = data[data['yymmdd'] <= self.date['evaluation']['to']]

        # Convert minus values to zeros
        data['pred'] = np.where(data['pred'] < 0, 0, data['pred'])  # Todo Exception

        return data

    def conv_data_type(self, data: pd.DataFrame):
        for col in self.str_type_cols:
            if col in list(data.columns):
                data[col] = data[col].astype(str)

        return data

    def add_item_info(self, data: pd.DataFrame):
        hrchy_item_list = self.hrchy['list']['item'][:self.hrchy['lvl']['item']]
        hrchy_item_nm_list = [self.hrchy_name_map[code] for code in hrchy_item_list]
        hrchy_item_all = hrchy_item_list + hrchy_item_nm_list
        hrchy_item_all = [self.item_name_map[col] for col in hrchy_item_all]
        merged = pd.merge(data, self.item_mst, on=[self.sku_col])

        return merged, hrchy_item_all

    def resample_sales(self, data: pd.DataFrame):
        # Add item info
        data, hrchy_item = self.add_item_info(data=data)
        data_grp = data.groupby(by=['division_cd'] + hrchy_item + ['cust_grp_cd', 'yy', 'week']).sum()
        data_grp = data_grp.reset_index()

        return data_grp, hrchy_item

    def fill_na_week(self, data):
        result = pd.DataFrame()
        date_len = len(self.pred_date_range)

        if (self.hrchy['lvl']['item'] == 5) or (self.lvl_cfg['middle_out']):
            cust_sku = data[self.key_col].drop_duplicates()

            for cust, sku in zip(cust_sku['cust_grp_cd'], cust_sku[self.sku_col]):
                temp = data[data['cust_grp_cd'] == cust]
                temp = temp[temp[self.sku_col] == sku]
                if sum(temp['sales'].isna()) != date_len:
                    result = pd.concat([result, temp])
        else:
            hrchy_key = self.item_name_map[self.hrchy['apply'][-1]]
            hrchy_list = list(data[hrchy_key].unique())
            for hrchy_code in hrchy_list:
                temp = data[data[hrchy_key] == hrchy_code]
                if sum(temp['sales'].isna()) != date_len:
                    result = pd.concat([result, temp])
            # if sum(temp['sales'].isna()) == 0:
            #     result = pd.concat([result, temp])

        return result
