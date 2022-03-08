import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import os
from typing import Dict, Tuple
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# font_path = r'C:\Windows\Fonts\malgun.ttf'
# font_name = fm.FontProperties(fname=font_path).get_name()
# matplotlib.rc('font', family='Malgun Gothic')
plt.style.use('fivethirtyeight')


class CalcAccuracy(object):
    item_cd_list = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd']
    col_fixed = ['division_cd', 'start_week_day', 'week']
    pick_sp1 = {
        # SP1: 도봉 / 안양 / 동울산 / 논산 / 동작 / 진주 / 이마트 / 롯데슈퍼 / 7-11
        'P1': ['1005', '1022', '1051', '1063', '1107', '1128', '1065', '1073', '1173', '1196'],
        'P2': ['1017', '1098', '1101', '1112', '1128', '1206', '1213', '1230']
    }
    pred_csv_map = {
        'name': {
            'C1-P3': 'pred_best.csv',
            'C1-P5': 'pred_middle_out_db.csv'
        },
        'encoding': {
            'C1-P3': 'cp949',
            'C1-P5': 'utf-8'
        }
    }

    def __init__(self, exec_kind: str, step_cfg: dict, exec_cfg: dict, date_cfg: dict, data_cfg: dict,
                 acc_classify_standard=0.25):
        # Object instance attribute
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.root_path = data_cfg['root_path']
        self.save_path = data_cfg['save_path']
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        # Execution instance attribute
        self.exec_kind = exec_kind
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg
        self.date_cfg = date_cfg
        self.data_cfg = data_cfg

        # Data instance attribute
        self.level = {}
        self.hrchy = {}
        self.division = data_cfg['division']
        self.cust_map = {}
        self.item_info = None
        self.date_sales = {}
        self.data_vrsn_cd = ''
        self.hist_date_range = []

        # Evaluation instance attribute
        self.acc_classify_standard = acc_classify_standard
        self.week_compare = 1             # Compare week range
        self.sales_threshold = 5          # Minimum sales quantity
        self.eval_threshold = 0.5         # Accuracy: Success or not
        self.filter_acc_rate = 2          # Filter accuracy rate
        self.filter_top_n_threshold = 1   # Filter top N
        self.filter_sales_threshold_standard = 'recent'    # hist / recent

    def run(self) -> None:
        # Initialize information
        self.init()

        # Load the dataset
        sales_compare, sales_hist, pred, plan = self.load_dataset()

        # Preprocess the dataset
        if self.step_cfg['cls_prep']:
            sales_hist, sales_compare, plan = self.preprocessing(
                sales_hist=sales_hist,
                sales_compare=sales_compare,
                plan=plan
            )

        # Filter forecast sku
        pred = self.filter_sku(pred=pred, plan=plan)

        # Compare result
        result = None
        if self.step_cfg['cls_comp']:
            result = self.compare_result(sales=sales_compare, pred=pred, plan=plan)

        # Change data type (string to datetime)
        # sales_hist = self.conv_to_datetime(data=sales_hist, col='start_week_day')
        # result = self.conv_to_datetime(data=result, col='start_week_day')

        if self.exec_cfg['pick_specific_biz_yn']:
            result = result[result['item_attr01_cd'] == self.data_cfg['item_attr01_cd']]

        # Accuracy rate bu SP1 + item level
        # if self.exec_cfg['calc_acc_by_sp1_item_yn']:
        #     self.calc_acc_by_sp1_line(data=result, sp1_list=self.pick_sp1[self.data_cfg['item_attr01_cd']])

        # Execute on top N accuracy
        if self.step_cfg['cls_top_n']:
            # Execute on specific sp1 list
            if self.exec_cfg['pick_specific_sp1_yn']:
                result_pick = self.pick_specific_sp1(data=result)
                for sp1, data in result_pick.items():
                    # Filter n by accuracy
                    data_filter_n = self.filter_n_by_accuracy(data=data)

                    # Draw plots
                    if self.step_cfg['cls_graph']:
                        self.draw_plot(result=data_filter_n, sales=sales_hist)

            else:
                # Filter n by accuracy
                result = self.filter_n_by_accuracy(data=result)

                # Draw plots
                if self.step_cfg['cls_graph']:
                    self.draw_plot(result=result, sales=sales_hist)

    # def calc_acc_by_sp1_line(self, data: pd.DataFrame, sp1_list=None) -> None:
    #     if sp1_list is not None:
    #         data = data[data['cust_grp_cd'].isin(sp1_list)]
    #
    #     grp_col = ['cust_grp_cd', 'item_attr01_cd', 'item_attr02_cd']
    #     grp_avg = data.groupby(by=grp_col)['success'].mean().reset_index()
    #     grp_avg_pivot = grp_avg.pivot(
    #         index=['item_attr01_cd', 'item_attr02_cd'],
    #         columns='cust_grp_cd',
    #         values='success')
    #
    #     path = os.path.join(self.save_path, self.data_vrsn_cd,
    #                         self.data_vrsn_cd + '_' + self.division + '_' + str(self.hrchy['lvl']['item']) +
    #                         '_' + self.data_cfg['item_attr01_cd'] + '_pivot.csv')
    #
    #     grp_avg_pivot.to_csv(path)

    def preprocessing(self, sales_hist: pd.DataFrame, sales_compare: pd.DataFrame, plan: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Resample data
        sales_compare = self.resample_sales(data=sales_compare)
        sales_hist = self.resample_sales(data=sales_hist)

        # remove zero quantity
        if self.exec_cfg['rm_zero_yn']:
            print("Remove zero sales quantities")
            sales_compare = self.filter_zeros(data=sales_compare, col='sales')

        if self.exec_cfg['filter_sales_threshold_yn']:
            print(f"Filter sales under {self.sales_threshold} quantity")
            sales_compare = self.filter_sales_threshold(hist=sales_hist, recent=sales_compare)

        # Fill na
        plan = plan.fillna(0)

        return sales_hist, sales_compare, plan

    def filter_sku(self, pred, plan) -> pd.DataFrame:
        mask_col = ['cust_grp_cd', 'item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd']
        plan_mask = plan[mask_col].drop_duplicates().copy()
        pred = pd.merge(pred, plan_mask, how='inner', on=mask_col)

        return pred

    def pick_specific_sp1(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        pick_sp1 = self.pick_sp1[self.data_cfg['item_attr01_cd']]
        data_pick = data[data['cust_grp_cd'].isin(pick_sp1)]

        # sorting
        data_pick = data_pick.sort_values(by=['cust_grp_cd', 'accuracy'])

        # split by sp1
        split = {}
        for sp1 in pick_sp1:
            temp = data_pick[data_pick['cust_grp_cd'] == sp1]
            if len(temp) > 0:
                split[sp1] = temp.reset_index(drop=True)

        return split

    def filter_sales_threshold(self, hist: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
        recent_filtered = None
        if self.filter_sales_threshold_standard == 'hist':
            hist_avg = self.calc_avg_sales(data=hist)
            hist_data_level = self.filter_avg_sales_by_threshold(data=hist_avg)
            recent_filtered = self.merged_filtered_data_level(data=recent, data_level=hist_data_level)

        elif self.filter_sales_threshold_standard == 'recent':
            recent_filtered = recent[recent['sales'] >= self.sales_threshold]

        return recent_filtered

    def calc_avg_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        avg = data.groupby(by=self.hrchy['apply']).mean()
        avg = avg.reset_index()

        return avg

    def filter_avg_sales_by_threshold(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data['sales'] >= self.sales_threshold]
        data = data[self.hrchy['apply']]    # Data level columns

        return data

    def merged_filtered_data_level(self, data: pd.DataFrame, data_level: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(data, data_level, on=self.hrchy['apply'], how='inner')

        return merged

    def init(self) -> None:
        self.set_date()         # Set date
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()        # Set the hierarchy
        self.get_item_info()    # Get item information
        self.set_cust_info()    # Get customer information
        self.make_dir()         # Make the directory

    def set_cust_info(self) -> None:
        # cust_info = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_grp_info())
        # cust_info['cust_grp_cd'] = cust_info['cust_grp_cd'].astype(str)
        # cust_info = cust_info.set_index('cust_grp_cd')['cust_grp_nm'].to_dict()
        cust_nm = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_nm_master())
        cust_nm['code'] = cust_nm['code'].astype(str)
        cust_map = defaultdict(lambda: defaultdict(dict))
        for cust_type, code, name in zip(cust_nm['type'], cust_nm['code'], cust_nm['name']):
            cust_map[cust_type][code] = name
        # cust_nm = cust_nm.set_index('code')['name'].to_dict()

        self.cust_map = cust_map

    def get_item_info(self) -> None:
        # Get the item master dataset
        item_info = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())
        item_info.columns = [config.HRCHY_CD_TO_DB_CD_MAP.get(col, col) for col in item_info.columns]
        item_info.columns = [config.HRCHY_SKU_TO_DB_SKU_MAP.get(col, col) for col in item_info.columns]

        # Change data type
        if 'item_cd' in item_info.columns:
            item_info['item_cd'] = item_info['item_cd'].astype(str)

        item_col_grp = self.hrchy['list']['item']
        item_col_grp = [[col, col[:-2] + 'nm'] for col in item_col_grp]
        item_col_list = []
        for col in item_col_grp:
            item_col_list.extend(col)

        item_info = item_info[item_col_list].drop_duplicates()

        # Add mege_yn information
        item_mega_info = self.io.get_df_from_db(sql=self.sql_conf.sql_item_mega_yn())
        item_info = pd.merge(item_info, item_mega_info, how='inner', on='item_cd')

        self.item_info = item_info

    def set_date(self) -> None:
        if self.date_cfg['cycle_yn']:
            self.date_sales = self.date_cfg['date']
            # self.date_sales = self.calc_sales_date()
            self.data_vrsn_cd = self.date_sales['hist']['from'] + '-' + self.date_sales['hist']['to']
            self.hist_date_range = pd.date_range(
                start=self.date_sales['hist']['from'],
                end=self.date_sales['hist']['to'],
                freq='w'
            )
        else:
            self.date_sales = self.date_cfg['date']
            self.data_vrsn_cd = self.date_cfg['data_vrsn_cd']
            self.hist_date_range = pd.date_range(
                start=self.date_sales['hist']['from'],
                end=self.date_sales['hist']['to'],
                freq='W-MON'
            )

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
                'item': [config.HRCHY_CD_TO_DB_CD_MAP.get(col, 'item_cd') for col
                         in self.common['hrchy_item'].split(',')[:self.level['item_lvl']]]
            }
        }
        self.hrchy['apply'] = self.hrchy['list']['cust'] + self.hrchy['list']['item']

    def make_dir(self) -> None:
        path = os.path.join(self.save_path, self.data_vrsn_cd)
        if not os.path.isdir(path):
            os.mkdir(path=path)
            if self.step_cfg['cls_graph']:
                os.mkdir(path=os.path.join(path, 'plot'))

    def calc_sales_date(self) -> Dict[str, Dict[str, str]]:
        today = datetime.date.today()
        today = today - datetime.timedelta(days=today.weekday())

        # History dates
        hist_from = today - datetime.timedelta(days=int(self.common['week_hist']) * 7 + 7)
        hist_to = today - datetime.timedelta(days=1 + 7)

        hist_from = hist_from.strftime('%Y%m%d')
        hist_to = hist_to.strftime('%Y%m%d')

        # Compare dates
        compare_from = today - datetime.timedelta(days=self.week_compare * 7)
        compare_to = today - datetime.timedelta(days=1)

        compare_from = compare_from.strftime('%Y%m%d')
        compare_to = compare_to.strftime('%Y%m%d')

        date = {
            'hist': {
                'from': hist_from,
                'to': hist_to
            },
            'compare': {
                'from': compare_from,
                'to': compare_to
            }
        }

        return date

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # load sales dataset
        info_sales_compare = {
            'division_cd': self.division,
            'start_week_day': self.date_sales['compare']['from']
        }
        sales_compare = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales_compare = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales_compare = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))

        info_sales_hist = {
            'division_cd': self.division,
            'from': self.date_sales['hist']['from'],
            'to': self.date_sales['hist']['to']
        }
        sales_hist = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales_hist = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales_hist = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))

        # Load prediction dataset
        pred = None
        if self.data_cfg['load_option'] == 'db':
            info_pred = {
                'data_vrsn_cd': self.data_vrsn_cd,
                'division_cd': self.division,
                'fkey': self.hrchy['key'][:-1],
                'yymmdd': self.date_sales['compare']['from'],
            }
            pred = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_best(**info_pred))

        elif self.data_cfg['load_option'] == 'csv':
            path = os.path.join(self.root_path, 'prediction', self.exec_kind, self.data_vrsn_cd,
                                self.division + '_' + self.data_vrsn_cd + '_C1-P3-'
                                + self.pred_csv_map['name'][self.hrchy['key'][:-1]])
            pred = pd.read_csv(path, encoding=self.pred_csv_map['encoding'][self.hrchy['key'][:-1]])
            pred.columns = [col.lower() for col in pred.columns]
            pred = pred.rename(columns={'yymmdd': 'start_week_day', 'result_sales': 'pred'})
            pred['cust_grp_cd'] = pred['cust_grp_cd'].astype(str)
            pred['start_week_day'] = pred['start_week_day'].astype(str)
            if 'item_cd' in pred.columns:
                pred['item_cd'] = pred['item_cd'].astype(str)

        pred = self.filter_col(data=pred)

        # Load plan dataset
        info_plan = {'yymmdd': self.date_sales['compare']['from']}
        plan = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_plan_confirm(**info_plan))

        return sales_compare, sales_hist, pred, plan

    def filter_col(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_col = self.hrchy['apply'] + self.col_fixed + ['pred']
        data = data[filter_col]

        return data

    def resample_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        grp_col = self.hrchy['apply'] + self.col_fixed
        data = data.groupby(by=grp_col).sum()
        data = data.reset_index()

        return data

    @staticmethod
    def filter_zeros(data: pd.DataFrame, col: str) -> pd.DataFrame:
        return data[data[col] != 0]

    def fill_missing_date(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        idx_add = list(set(self.hist_date_range) - set(df[col]))
        data_add = np.zeros(len(idx_add))
        # df_add = pd.DataFrame(data_add, index=idx_add, columns=df.columns)
        df_add = pd.DataFrame(np.array([idx_add, data_add]).T, columns=df.columns)
        df = df.append(df_add)
        df = df.sort_values(by=col)

        return df

    def merge_result(self, dividend: pd.DataFrame, divisor: pd.DataFrame, merge_col: list) -> pd.DataFrame:
        merged = pd.merge(
            divisor,
            dividend,
            on=merge_col,
            how='inner'    # Todo: choose left or inner
        )
        merged = merged.fillna(0)

        return merged

    def compare_result(self, sales: pd.DataFrame, pred: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
        fcst_accuracy = self.calc_fcst_accuracy(dividend=sales, divisor=pred, kind='fcst')
        plan_accuracy = self.calc_plan_accuracy(dividend=sales, divisor=plan, kind='plan')

        merged_col = ['cust_grp_cd'] + self.item_cd_list
        merged = pd.merge(
            plan_accuracy, fcst_accuracy, how='left', on=merged_col, suffixes=('', '_DROP')
            # plan_accuracy, fcst_accuracy, how = 'inner', on = merged_col, suffixes = ('', '_DROP')
        ).filter(regex='^(?!.*_DROP)')

        # Add name information
        merged = self.add_information(data=merged)

        # Save result on db
        if self.exec_cfg['save_db_yn']:
            self.save_result_on_db(data=merged)

        # Save the result to file
        self.save_result_to_file(data=merged)

        return merged

    def save_result_on_db(self, data):
        #
        data_db = data.copy()
        data_db = data_db.rename(columns={
            'cust_grp_cd': 'sp1_cd',
            'cust_grp_nm': 'sales_mgmt_nm',
        })

        data_db['project_cd'] = 'ENT001'
        data_db['data_vrsn_cd'] = self.data_vrsn_cd
        data_db['division_cd'] = self.division
        data_db['yymmdd'] = self.date_cfg['date']['compare']['from']
        data_db['sales_mgmt_cd'] = data_db['sp2_c_cd'] + data_db['sp2_cd'] + data_db['sp1_c_cd'] + data_db['sp1_cd']

        self.io.insert_to_db(df=data_db, tb_name='M4S_O110630')

    def save_result_to_file(self, data: pd.DataFrame) -> None:
        # Reorder and filter columns
        reordered = self.reorder_filter_column(data=data)

        # Save the result
        path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '_' + str(self.acc_classify_standard) + '.csv')
        reordered.to_csv(path, index=False, encoding='cp949')

    def calc_fcst_accuracy(self, dividend: pd.DataFrame, divisor: pd.DataFrame, kind: str):
        # Merge dataset
        merge_col = self.hrchy['apply'] + self.col_fixed
        data = self.merge_result(dividend=dividend, divisor=divisor, merge_col=merge_col)

        # Calculate the accuracy
        data = self.calc_accuracy(data=data, dividend='sales', divisor='pred')

        data = self.classify_accuracy(data=data, kind=kind)

        grp_col = ['cust_grp_cd'] + self.item_cd_list
        data = self.eval_result_dev(data=data, grp_col=grp_col, kind=kind)
        data = data.reset_index()

        return data

    def calc_plan_accuracy(self, dividend: pd.DataFrame, divisor: pd.DataFrame, kind: str):
        # Merge dataset
        merge_col = self.hrchy['apply'] + ['start_week_day', 'week']
        data = self.merge_result(dividend=dividend, divisor=divisor, merge_col=merge_col)

        #
        mega = data.groupby(by=['item_attr03_cd', 'mega_yn'])['item_cd'].count().reset_index()
        mega = mega[['item_attr03_cd', 'mega_yn']].copy()

        # Calculate the accuracy
        data = self.calc_accuracy(data=data, dividend='sales', divisor='planed')

        # Classify the accuracy
        data = self.classify_accuracy(data=data, kind=kind)

        grp_col = ['sp2_c_cd', 'sp2_cd', 'sp1_c_cd', 'cust_grp_cd'] + self.item_cd_list
        data = self.eval_result_dev(data=data, grp_col=grp_col, kind=kind)
        data = data.reset_index()

        # Merge mega information
        data = pd.merge(data, mega, how='left', on='item_attr03_cd')

        return data

    def add_information(self, data):
        # Add naming
        data['sp2_c_nm'] = [self.cust_map['SP2_C'].get(code, code) for code in data['sp2_c_cd'].values]
        data['sp2_nm'] = [self.cust_map['SP2'].get(code, code) for code in data['sp2_cd'].values]
        data['sp1_c_nm'] = [self.cust_map['SP1_C'].get(code, code) for code in data['sp1_c_cd'].values]
        data['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in data['cust_grp_cd'].values]

        # Add item names
        item_info = self.item_info[self.item_cd_list + [code[:-2] + 'nm' for code in self.item_cd_list]]\
            .drop_duplicates()\
            .copy()
        data = pd.merge(data, item_info, how='left', on=self.item_cd_list)

        # view_col = ['cust_grp_cd', 'cust_grp_nm'] + list(self.item_info.columns) + ['sales', 'pred', 'accuracy']
        # data = data[view_col]

        return data

    def reorder_filter_column(self, data: pd.DataFrame):
        cust = ['sp2_c_nm', 'sp2_nm', 'sp1_c_nm', 'cust_grp_nm']
        item = ['item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'mega_yn']
        fcst = ['level_cnt', 'cover_fcst_cnt', 'less_fcst_cnt', 'over_fcst_cnt', 'zero_fcst_cnt',
                'cover_fcst_rate', 'less_fcst_rate', 'over_fcst_rate', 'zero_fcst_rate']
        plan = ['cover_plan_cnt', 'less_plan_cnt', 'over_plan_cnt', 'zero_plan_cnt',
                'cover_plan_rate', 'less_plan_rate', 'over_plan_rate', 'zero_plan_rate']

        data_reorder = data[cust + item + fcst + plan].copy()

        return data_reorder

    def classify_accuracy(self, data: pd.DataFrame, kind: str) -> pd.DataFrame:
        condition = [
            data['sales'] == 0,
            data['accuracy'] < 1 - self.acc_classify_standard,
            data['accuracy'] > 1 + self.acc_classify_standard
        ]
        label = kind + '_cnt'
        values = ['zero_' + label, 'less_' + label, 'over_' + label]
        data['classification'] = np.select(condlist=condition, choicelist=values, default=None)
        data['classification'] = data['classification'].fillna('cover_' + label)

        return data

    def eval_result_dev(self, data: pd.DataFrame, grp_col: list, kind: str) -> pd.DataFrame:
        level_cnt = data.groupby(by=grp_col)['item_cd']\
            .count()\
            .rename('level_cnt')

        classify_cnt = data.groupby(by=grp_col + ['classification'])['item_cd']\
            .count()\
            .astype(int)\
            .reset_index()\
            .rename(columns={'item_cd': 'classify_cnt'})

        classify_cnt = classify_cnt.pivot(
            index=grp_col,
            columns=['classification'],
            values='classify_cnt'
        ).fillna(0)

        tot_cnt = pd.merge(level_cnt, classify_cnt, left_index=True, right_index=True)

        if 'zero_' + kind + '_cnt' not in tot_cnt.columns:
            tot_cnt['zero_' + kind + '_cnt'] = 0

        # Calculate rates
        for classify_kind in ['cover', 'less', 'over', 'zero']:
            label = classify_kind + '_' + kind
            tot_cnt[label + '_rate'] = np.round(
                tot_cnt[label + '_cnt'] / tot_cnt['level_cnt'], 2)

        return tot_cnt

    @staticmethod
    def calc_accuracy(data, dividend: str, divisor: str) -> pd.DataFrame:
        conditions = [
            data[divisor] == 0,
            data[dividend] != data[divisor]
        ]
        # values = [1, 0, data['pred'] / data['sales']]
        values = [0, data[dividend] / data[divisor]]    # Todo: logic changed (22.02.09)
        data['accuracy'] = np.select(conditions, values)

        # customize accuracy
        # data = util.customize_accuracy(data=data, col='accuracy')

        return data

    def eval_result(self, data) -> pd.DataFrame:
        data['success'] = np.where(data['accuracy'] < self.eval_threshold, 0, 1)
        # data['success'] = np.where(data['accuracy'] < self.eval_threshold, 'N', 'Y')

        len_total = len(data)
        len_success = len(data[data['success'] == 1])
        len_fail = len(data[data['success'] == 0])

        print("---------------------------------")
        print(f"Prediction Total: {len_total}")
        print(f"Prediction Success: {len_success}, {round(len_success/len_total, 3)}")
        print(f"Prediction Fail: {len_fail}, {round(len_fail/len_total, 3)}")
        print("---------------------------------")

        return data

    def filter_n_by_accuracy(self, data) -> Dict[str, pd.DataFrame]:
        data = data.sort_values(by=['accuracy'], ascending=False)
        data = data[data['accuracy'] != 1]    # remove zero equal zero case
        best = data.iloc[:self.filter_top_n_threshold, :]
        worst = data.iloc[-self.filter_top_n_threshold:, :]
        worst = worst.fillna(0)

        result = {'best': best, 'worst': worst}

        return result

    def draw_plot(self, result: dict, sales: pd.DataFrame):
        item_lvl_col = self.hrchy['list']['item'][-1]
        plt.clf()
        for kind, data in result.items():
            for cust_grp, item_lvl_cd, date, pred, sale in zip(
                    data['cust_grp_cd'], data[item_lvl_col], data['start_week_day'], data['pred'], data['sales']
            ):

                filtered = sales[(sales['cust_grp_cd'] == cust_grp) & (sales[item_lvl_col] == item_lvl_cd)]
                filtered = filtered[['start_week_day', 'sales']]
                filtered = self.fill_missing_date(df=filtered, col='start_week_day')

                fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
                # fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                ax1.plot(filtered['start_week_day'], filtered['sales'],
                         linewidth=1.3, color='grey', label='history')

                if len(filtered) > 0:
                    df_sales = self.connect_dot(hist=filtered, date=date, value=sale)
                    df_pred = self.connect_dot(hist=filtered, date=date, value=pred)
                    ax1.plot(df_sales['start_week_day'], df_sales['qty'], color='darkblue',
                             linewidth=1.3, alpha=0.7)
                    ax1.plot(df_pred['start_week_day'], df_pred['qty'], color='firebrick',
                             linewidth=2, alpha=0.8, linestyle='dotted')

                    # Ticks
                    # plt.xlim(xmin=df_sales['start_week_day'].tolist()[0],
                    #          xmax=df_sales['start_week_day'].tolist()[-1])
                    plt.xticks(rotation=30)
                    plt.ylim(ymin=-5)
                    date_str = datetime.datetime.strftime(date, '%Y%m%d')
                    plt.title(f'Date: {date_str} / Cust Group: {cust_grp} / Brand: {self.item_info.get(item_lvl_cd, item_lvl_cd)}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        self.root_path, 'analysis', 'accuracy',
                        self.data_vrsn_cd, 'plot', self.data_vrsn_cd + '_' + self.division + '_' +
                        kind + '_' + str(cust_grp) + '_' + str(item_lvl_cd) + '.png'))

                    plt.close(fig)

    @staticmethod
    def connect_dot(hist: pd.DataFrame, date, value) -> pd.DataFrame:
        hist = hist[['start_week_day', 'sales']]
        hist = hist.rename(columns={'sales': 'qty'})
        hist = hist.sort_values(by='start_week_day')

        result = pd.DataFrame({'start_week_day': [date], 'qty': [value]})
        result = result.append(hist.iloc[-1, :], ignore_index=True)

        return result

    @staticmethod
    def conv_to_datetime(data: pd.DataFrame, col: str) -> pd.DataFrame:
        data[col] = pd.to_datetime(data[col])

        return data
