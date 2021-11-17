import common.util as util

import numpy as np
import pandas as pd


class ResultReport(object):
    drop_col1 = ['project_cd', 'data_vrsn_cd', 'fkey', 'create_user_cd']
    drop_col2 = ['division_cd', 'biz_cd', 'line_cd', 'brand_cd', 'item_cd']
    summary_col = {
        'C1-P5': ['cust_grp_nm', 'biz_nm', 'line_nm', 'brand_nm', 'item_nm', 'sku_cd', 'sku_nm'],
        'C0-P4': ['biz_nm', 'line_nm', 'brand_nm', 'item_nm'],
        'C0-P3': ['biz_nm', 'line_nm', 'brand_nm']
    }
    hrchy_name_map = {'biz_cd': 'biz_nm', 'line_cd': 'line_nm', 'brand_cd': 'brand_nm', 'item_cd': 'item_nm'}

    item_name_map = {
        'item_attr01_cd': 'biz_cd', 'item_attr02_cd': 'line_cd', 'item_attr03_cd': 'brand_cd',
        'item_attr04_cd': 'item_cd', 'item_attr01_nm': 'biz_nm', 'item_attr02_nm': 'line_nm',
        'item_attr03_nm': 'brand_nm', 'item_attr04_nm': 'item_nm'
    }

    def __init__(self, common, division, data_vrsn, hrchy, cust_grp_mst, item_mst, cal_mst):
        self.division = division
        self.common = common
        self.hrchy = hrchy
        self.cal_mst = cal_mst
        self.item_mst = item_mst
        self.cust_grp_mst = cust_grp_mst
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
        summary = self.make_summary(df=raw_all)

    def group_by_week(self, data: pd.DataFrame):
        pass

    def make_raw_result(self, sales, pred):
        hrchy_item = None
        if self.hrchy['key'][:-1] != 'C1-P5':
            sales, hrchy_item = self.resample_sales(data=sales)

        # Preprocess the prediction dataset
        pred = pred.rename(columns={'result_sales': 'pred', 'item_cd': 'sku_cd', 'item_nm': 'sku_nm'})
        pred = pred.rename(columns=self.item_name_map)
        pred = pred.drop(columns=self.drop_col1, errors='ignore')
        pred['pred'] = np.round(pred['pred'].to_numpy(), 2)
        pred['yy'] = pred[self.common['date_col']].str.slice(0, 4)

        # Filter comparing days
        pred = pred[pred['yymmdd'] >= self.common['pred_start_day']]
        pred = pred[pred['yymmdd'] <= self.common['pred_end_day']]

        pred['pred'] = np.where(pred['pred'] < 0, 0, pred['pred'])  # Todo Exception

        if self.hrchy['key'][:-1] == 'C1-P5':
            merged = pd.merge(
                pred,
                sales,
                on=['division_cd', 'yy', 'week', 'cust_grp_cd', 'sku_cd'],
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

        else:
            merged = pd.merge(
                pred,
                sales,
                on=['division_cd', 'yy', 'week'] + hrchy_item,
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

        # Fill NA weeks with 0 qty
        merged = self.fill_na_week(data=merged)

        # Drop dates that doesn't compare
        merged = merged.fillna(0)

        # Calculate absolute difference
        merged['diff'] = merged['sales'] - merged['pred']
        merged['diff'] = np.absolute(merged['diff'].to_numpy())

        # Add information
        # if (self.hrchy['key'][:-1] == 'C1-P5') or (self.hrchy['key'][:-1] == 'C0-P5'):
        #     item_mst = self.item_mst[['sku_cd', 'sku_nm']]
        #     merged = pd.merge(merged, item_mst, on='sku_cd')

        # Drop unnecessary columns
        merged = merged.drop(columns=self.drop_col2, errors='ignore')

        # Sort columns
        if self.hrchy['key'][:-1] == 'C1-P5':
            merged = merged[['cust_grp_nm', 'biz_nm', 'line_nm', 'brand_nm', 'item_nm', 'sku_cd', 'sku_nm',
                             'yy', 'week', 'stat_cd', 'sales', 'pred', 'diff']]
        else:
            hrchy_name = [self.hrchy_name_map[code] for code in self.hrchy['apply']]
            merged = merged[hrchy_name + ['yy', 'week', 'sales', 'pred', 'diff']]

        # Save the result
        merged.to_csv(self.save_path['all'], index=False, encoding='CP949')
        # merged['diff_abs'] = np.absolute(merged['diff'].to_numpy())

        return merged

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
        if self.hrchy['key'][:-1] == 'C1-P5':
            cust_sku = data[['cust_grp_cd', 'sku_cd']].drop_duplicates()

            for cust, sku in zip(cust_sku['cust_grp_cd'], cust_sku['sku_cd']):
                temp = data[data['cust_grp_cd'] == cust]
                temp = temp[temp['sku_cd'] == sku]
                if sum(temp['sales'].isna()) != date_len:
                    result = pd.concat([result, temp])
        else:
            hrchy_key = self.hrchy['apply'][-1]
            hrchy_list = list(data[hrchy_key].unique())
            for hrchy_code in hrchy_list:
                temp = data[data[hrchy_key] == hrchy_code]
                if sum(temp['sales'].isna()) != date_len:
                    result = pd.concat([result, temp])
            # Exception
            # if sum(temp['sales'].isna()) == 0:
            #     result = pd.concat([result, temp])

        return result

    def make_summary(self, df):
        df = df.drop(columns=['diff'])

        # group by sum
        summary_sum = df.groupby(by=self.summary_col[self.hrchy['key'][:-1]]).sum()
        # group by mean
        summary_mean = df.groupby(by=self.summary_col[self.hrchy['key'][:-1]]).mean()

        # Rename columns
        summary_sum = summary_sum.rename(columns={
            'sales': 'sales_sum',
            'pred': 'pred_sum'
        })
        summary_mean = summary_mean.rename(columns={
            'sales': 'sales_mean',
            'pred': 'pred_mean'
        })

        summary = pd.merge(summary_sum, summary_mean, left_index=True, right_index=True)
        summary = summary.reset_index()

        # Calculate accuracy
        summary['accuracy_sum'] = np.round(summary['pred_sum'] / summary['sales_sum'], 2) * 100
        summary['accuracy_mean'] = np.round(summary['pred_mean'] / summary['sales_mean'], 2) * 100

        result_cols = ['sales_sum', 'pred_sum', 'accuracy_sum', 'sales_mean', 'pred_mean', 'accuracy_mean']
        summary = summary[self.summary_col[self.hrchy['key'][:-1]] + result_cols]

        summary.to_csv(self.save_path['summary'], index=False, encoding='CP949')

        summary_score = summary[['accuracy_sum', 'accuracy_mean']].mean()
        summary_score = np.round(summary_score, 2)
        print(f"Accuracy Sum: {summary_score[0]}")
        print(f"Accuracy Mean: {summary_score[1]}")

        return summary
