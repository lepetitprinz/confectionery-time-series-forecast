import common.util as util

import numpy as np
import pandas as pd


class ResultReport(object):
    drop_col = ['stat_cd', 'fkey', 'create_user_cd', 'division_cd', 'data_vrsn_cd',
                'item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd']
    summary_col = ['cust_grp_cd', 'cust_grp_nm', 'item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'item_attr04_nm',
                   'sku_cd', 'sku_nm']

    def __init__(self, common, division, data_vrsn, hrchy, cust_grp_mst, item_mst):
        self.division = division
        self.common = common
        self.cust_grp_mst = cust_grp_mst
        self.item_mst = item_mst
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

    def make_raw_result(self, sales, pred):
        pred = pred.rename(columns={'result_sales': 'pred', 'item_cd': 'sku_cd'})
        pred = pred.drop(columns=['project_cd', 'cust_grp_nm'])
        pred['pred'] = np.round(pred['pred'].to_numpy(), 2)

        pred['yy'] = pred[self.common['date_col']].str.slice(0, 4)
        merged = pd.merge(
            pred,
            sales,
            on=['division_cd', 'yy', 'week', 'cust_grp_cd', 'sku_cd'],
            how='left',
            suffixes=('', '_DROP')
        ).filter(regex='^(?!.*_DROP)')

        # Drop dates that doesn't compare
        merged = merged.dropna()

        merged['diff'] = merged['sales'] - merged['pred']

        item_mst = self.item_mst[['sku_cd', 'sku_nm']]

        # Add information
        merged = pd.merge(merged, item_mst, on='sku_cd')
        merged = pd.merge(merged, self.cust_grp_mst, on='cust_grp_cd')

        # Drop unnecessary columns
        merged = merged.drop(columns=self.drop_col)

        merged = merged[['cust_grp_cd', 'cust_grp_nm', 'item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm',
                         'item_attr04_nm', 'sku_cd', 'sku_nm', 'yy', 'week', 'sales', 'pred', 'diff']]

        merged.to_csv(self.save_path['all'], index=False, encoding='CP949')

        merged['diff_abs'] = np.absolute(merged['diff'].to_numpy())

        return merged

    def make_summary(self, df):
        summary_mean = df.groupby(by=self.summary_col).mean()
        summary_std = df.groupby(by=self.summary_col).std()
        summary_mean = summary_mean.drop(columns=['diff'])
        summary_std = summary_std.drop(columns=['diff'])

        summary_mean = summary_mean.rename(columns={
            'sales': 'sales_mean',
             'pred': 'pred_mean',
             'diff_abs': 'diff_abs_mean'
         })
        summary_std = summary_std.rename(columns={
            'sales': 'sales_std',
            'pred': 'pred_std',
            'diff_abs': 'diff_abs_std'
        })

        summary = pd.merge(summary_mean, summary_std, left_index=True, right_index=True)
        summary = summary.reset_index()
        summary = summary[['cust_grp_cd', 'cust_grp_nm', 'item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm',
                           'item_attr04_nm', 'sku_cd', 'sku_nm', 'sales_mean', 'pred_mean', 'diff_abs_mean',
                           'sales_std', 'pred_std', 'diff_abs_std']]

        summary.to_csv(self.save_path['summary'], index=False, encoding='CP949')

        summary_score = summary[['diff_abs_mean', 'diff_abs_std']].mean()
        summary_score = np.round(summary_score, 2)
        print(f"Absolute average of difference: {summary_score[0]}")
        print(f"Absolute standard deviation of difference: {summary_score[1]}")

        return summary
