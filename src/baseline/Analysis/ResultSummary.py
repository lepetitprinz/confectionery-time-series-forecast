import common.util as util

import numpy as np
import pandas as pd


class ResultSummary(object):
    drop_col1 = ['project_cd', 'data_vrsn_cd', 'fkey', 'create_user_cd']
    hrchy_name_map = {'biz_cd': 'biz_nm', 'line_cd': 'line_nm', 'brand_cd': 'brand_nm', 'item_cd': 'item_nm'}

    item_name_map = {
        'item_attr01_cd': 'biz_cd', 'item_attr02_cd': 'line_cd', 'item_attr03_cd': 'brand_cd',
        'item_attr04_cd': 'item_cd', 'item_attr01_nm': 'biz_nm', 'item_attr02_nm': 'line_nm',
        'item_attr03_nm': 'brand_nm', 'item_attr04_nm': 'item_nm'
    }
    item_name_rev_map = {val: key for key, val in item_name_map.items()}

    def __init__(self, common: dict, division: str, data_vrsn: str, test_vrsn: str,
                 hrchy: dict, item_mst: pd.DataFrame):
        # Data Information Configuration
        self.common = common
        self.item_mst = item_mst
        self.division = division
        self.data_vrsn = data_vrsn
        self.test_vrsn = test_vrsn

        # Data Level Configuration
        self.hrchy = hrchy
        self.hrchy_item_cd_list = common['db_hrchy_item_cd'].split(',')
        self.hrchy_item_nm_list = common['db_hrchy_item_nm'].split(',')

        self.str_type_cols = ['cust_grp_cd', 'sku_cd', self.common['date_col']]
        self.pred_date_range = pd.date_range(
            start=common['pred_start_day'],
            end=common['pred_end_day'],
            freq=common['resample_rule']
        )
        self.save_path = {
            'all': util.make_path_baseline(
                module='report', division=division, data_vrsn=data_vrsn, hrchy_lvl=hrchy['key'],
                step='all', extension='csv'),
            'summary': util.make_path_baseline(
                module='report', division=division, data_vrsn=data_vrsn, hrchy_lvl=hrchy['key'],
                step='summary', extension='csv')
        }

    def compare_result(self, sales, pred):
        raw_all = self.make_raw_result(sales=sales, pred=pred)
        # self.make_summary(df=raw_all)

        return raw_all

    def make_db_format(self, data):
        data['seq'] = [self.test_vrsn + '_' + str(i+1).zfill(7) for i in range(len(data))]
        data['project_cd'] = self.common['project_cd']
        data['data_vrsn_cd'] = self.data_vrsn
        data['division_cd'] = self.division
        data['test_vrsn_cd'] = self.test_vrsn
        data['fkey'] = 'C1-P5'
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

    def make_raw_result(self, sales, pred):
        hrchy_item = None
        # if self.hrchy['key'][:-1] != 'C1-P5':
        #     sales, hrchy_item = self.resample_sales(data=sales)

        # If execute middle out
        if self.hrchy['lvl']['item'] < 5:
            pred = self.conv_data_type(data=pred)
            pred['division_cd'] = self.division
            item_mst = self.item_mst.rename(columns=self.item_name_rev_map)
            pred = pd.merge(
                pred, item_mst, how='left', on=self.hrchy_item_cd_list[:-1] + ['sku_cd'],
                suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        pred = self.prep_prediction(data=pred)

        merged = pd.merge(
            pred,
            sales,
            on=['division_cd', 'yy', 'week', 'cust_grp_cd', 'sku_cd'],
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

        merged = merged.rename(columns=self.item_name_rev_map)
        merged = merged.rename(columns={'sku_cd': 'item_cd', 'sku_nm': 'item_nm'})

        # Sort columns
        hrchy_col = []
        fixed_col = ['stat_cd', 'yy', 'week', 'sales', 'pred', 'diff']
        if self.hrchy['key'][:2] == 'C1':
            hrchy_col.extend(['cust_grp_cd', 'cust_grp_nm'])

        hrchy_item = [[code, name] for code, name in zip(self.hrchy_item_cd_list, self.hrchy_item_nm_list)]
        for code, name in hrchy_item:
            hrchy_col.append(code)
            hrchy_col.append(name)

        merged = merged[hrchy_col + fixed_col]

        # Save the result
        # merged.to_csv(self.save_path['all'], index=False, encoding='CP949')

        return merged

    def prep_prediction(self, data):
        # Preprocess the prediction dataset

        # Rename columns
        data = data.rename(columns={'sales': 'pred', 'result_sales': 'pred', 'item_cd': 'sku_cd', 'item_nm': 'sku_nm'})
        data = data.rename(columns=self.item_name_map)
        data = data.drop(columns=self.drop_col1, errors='ignore')

        # Round prediction results
        data['pred'] = np.round(data['pred'].to_numpy(), 2)
        data['yy'] = data[self.common['date_col']].str.slice(0, 4)

        # Filter comparing days
        data = data[data['yymmdd'] >= self.common['pred_start_day']]
        data = data[data['yymmdd'] <= self.common['pred_end_day']]

        # Convert minus values to zeros
        data['pred'] = np.where(data['pred'] < 0, 0, data['pred'])  # Todo Exception

        return data

    def conv_data_type(self, data: pd.DataFrame):
        for col in self.str_type_cols:
            data[col] = data[col].astype(str)

        return data

    def add_item_info(self, data: pd.DataFrame):
        hrchy_item_list = self.hrchy['list']['item'][:self.hrchy['lvl']['item']]
        hrchy_item_nm_list = [self.hrchy_name_map[code] for code in hrchy_item_list]
        hrchy_item_all = hrchy_item_list + hrchy_item_nm_list
        item_info = self.item_mst[hrchy_item_all + ['sku_cd']]
        merged = pd.merge(data, item_info, on=['sku_cd'])

        return merged, hrchy_item_all

    def resample_sales(self, data: pd.DataFrame):
        # Add item info
        data, hrchy_item = self.add_item_info(data=data)
        data_grp = data.groupby(by=['division_cd'] + hrchy_item + ['yy', 'week']).sum()
        data_grp = data_grp.reset_index()

        return data_grp, hrchy_item

    def fill_na_week(self, data):
        result = pd.DataFrame()
        date_len = len(self.pred_date_range)

        # if self.hrchy['key'][:-1] == 'C1-P5':
        cust_sku = data[['cust_grp_cd', 'sku_cd']].drop_duplicates()

        for cust, sku in zip(cust_sku['cust_grp_cd'], cust_sku['sku_cd']):
            temp = data[data['cust_grp_cd'] == cust]
            temp = temp[temp['sku_cd'] == sku]
            if sum(temp['sales'].isna()) != date_len:
                result = pd.concat([result, temp])
        # else:
        #     hrchy_key = self.hrchy['apply'][-1]
        #     hrchy_list = list(data[hrchy_key].unique())
        #     for hrchy_code in hrchy_list:
        #         temp = data[data[hrchy_key] == hrchy_code]
        #         if sum(temp['sales'].isna()) != date_len:
        #             result = pd.concat([result, temp])
            # Exception
            # if sum(temp['sales'].isna()) == 0:
            #     result = pd.concat([result, temp])

        return result