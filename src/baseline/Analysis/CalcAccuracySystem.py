import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from collections import defaultdict


class CalcAccuracySystem(object):
    # all of item & date list
    item_list = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd']
    date_list = ['start_week_day', 'week']

    # Batch column list
    cust_batch_list = ['sp2_c_cd', 'sp2_cd', 'sp1_c_cd', 'cust_grp_cd']
    item_batch_list = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd']

    # Accuracy class
    classify_kind = ['cover', 'less', 'over', 'zero']
    item_lvl_map = {3: 'BRAND', 5: 'SKU'}

    # Customer mapping
    sp1_hrchy_map = {
        'SELL_IN': {
            '101': '1.시판',
            '102': '2.유통',
            '103': '3.EC',
            '107': '4.글로벌',
            '108': '4.글로벌',
            '109': '4.글로벌',
            '110': '4.글로벌',
        },
        'SELL_OUT': {
            '1065': '1.할인점',    # 이마트
            '1066': '1.할인점',    # 롯데마트
            '1067': '1.할인점',    # 홈플러스
            '1073': '2.유통점',    # 롯데슈퍼
            '1074': '2.유통점',    # GS유통
            '1075': '2.유통점',    # 홈프러스슈퍼
            '1076': '2.유통점',    # 이마트슈퍼
        }
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

    def __init__(self, exec_kind: str, exec_cfg: dict, date_cfg: dict, data_cfg: dict,
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
        self.exec_cfg = exec_cfg
        self.date_cfg = date_cfg
        self.data_cfg = data_cfg

        # Master table data instance attribute
        self.cal_mst = None
        self.item_mst = None
        self.sales_matrix = None

        # Data instance attribute
        self.level = {}
        self.hrchy = {}
        self.division = data_cfg['division']
        self.cust_map = {}
        self.date_sales = {}
        self.data_vrsn_cd = ''
        self.hist_date_range = []

        # Evaluation instance attribute
        self.week_compare = 1    # Compare week range
        self.acc_classify_standard = acc_classify_standard
        self.load_sales_option = 'fixed'    # fixed / recent

    def run(self) -> None:
        # Initialize information
        self.init()

        # Run batch process
        if self.exec_kind == 'batch':
            self.exec_batch()
        # Run dev process
        elif (self.exec_kind == 'dev') or (self.exec_kind == 'test'):
            self.exec_dev()

    # Execute the batch process
    def exec_batch(self) -> None:
        # Load the dataset
        pred_plan = self.load_data_batch()

        # Update recent sales matrix
        pred_plan = pd.merge(pred_plan, self.sales_matrix, how='inner', on=['cust_grp_cd', 'item_cd'])

        # Filter business range (Fixed)
        pred_plan = pred_plan[pred_plan['item_attr01_cd'] == 'P1']

        # Fill empty brand code
        pred_plan['item_attr03_cd'] = pred_plan['item_attr03_cd'].fillna('UNDEFINED')

        # Make the raw result
        data_raw = self.calc_raw(data=pred_plan)

        # Make the summary result
        # Summary of all brand
        self.calc_summary(data=data_raw, mega_filter=False)
        # Summary of mega brand
        self.calc_summary(data=data_raw, mega_filter=True)

        # Make the db result
        self.calc_db(data=data_raw)

    def exec_dev(self) -> None:
        # Load the dataset
        sales, pred = self.load_data_dev()

        # Preprocess the dataset
        merged = self.preprocess_dev(sales=sales, pred=pred)

        # Update recent sales matrix
        merged = pd.merge(merged, self.sales_matrix, how='inner', on=['cust_grp_cd', 'item_cd'])

        # Filter business range
        merged = merged[merged['item_attr01_cd'] == 'P1']

        # Fill empty brand code
        merged['item_attr03_cd'] = merged['item_attr03_cd'].fillna('UNDEFINED')

        self.calc_dev(data=merged)

    def preprocess_dev(self, sales, pred) -> pd.DataFrame:
        # Filter columns
        comm_col = ['cust_grp_cd'] + self.item_list + self.date_list
        pred = pred[comm_col + ['pred']]

        # Merge dataset
        merged = pd.merge(sales, pred, how='left', on=comm_col)
        merged = merged.fillna(0)

        return merged

    def calc_dev(self, data: pd.DataFrame) -> None:
        acc = self.calc_accuracy(data=data, dividend='sales', divisor='pred', label='_pred')

        # Add naming information
        acc['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in acc['cust_grp_cd'].values]

        # Add naming information
        merge_col = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd']
        acc = pd.merge(acc, self.item_mst, how='left', on=merge_col)
        acc['item_attr01_nm'] = acc['item_attr01_nm'].fillna('건과')

        # Reorder columns
        col_cust = ['cust_grp_nm']
        col_item = ['item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'item_attr04_nm',
                    'item_cd', 'item_nm', 'mega_yn']
        col_etc = ['sales', 'pred', 'acc_pred']
        acc = acc[col_cust + col_item + col_etc]

        # Save the result
        path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '_dev.csv')
        acc.to_csv(path, index=False, encoding='cp949')

    def calc_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        data_pred = self.calc_accuracy(data=data, dividend='sales', divisor='pred', label='_pred')
        data_plan = self.calc_accuracy(data=data, dividend='sales', divisor='plan', label='_plan')

        # Merge the calculation result of prediction & plan
        grp_col = self.cust_batch_list + self.hrchy['list']['item'] + \
                  ['mega_yn', 'start_week_day', 'week', 'sales', 'pred', 'plan']
        merged = pd.merge(data_pred, data_plan, on=grp_col)

        if self.division == 'SELL_IN':
            merged = self.calc_acc_exception(data=merged)

        # Add naming information
        result = merged.copy()
        result['sp1_c_nm'] = [self.cust_map['SP1_C'].get(code, code) for code in result['sp1_c_cd'].values]
        result['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in result['cust_grp_cd'].values]

        item_mst = self.item_mst.copy()
        item_mst = item_mst[['item_cd', 'item_nm', 'item_attr01_nm', 'item_attr02_nm',
                             'item_attr03_nm', 'item_attr04_nm']]
        # merge_col = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd', 'mega_yn']
        result = pd.merge(result, item_mst, how='left', on=['item_cd'])
        result['item_attr01_nm'] = result['item_attr01_nm'].fillna('건과')

        # Reorder columns
        col_cust = ['sp1_c_nm', 'cust_grp_nm']
        col_item = ['item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'item_attr04_nm',
                    'item_cd', 'item_nm', 'mega_yn']
        col_etc = ['sales', 'pred', 'plan', 'acc_pred', 'acc_plan']
        result = result[col_cust + col_item + col_etc]

        # Save the result
        if self.exec_cfg['save_file_yn']:
            path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division +
                                '_' + str(self.hrchy['lvl']['item']) + '_raw.csv')
            result.to_csv(path, index=False, encoding='cp949')

        return merged

    @staticmethod
    def calc_acc_exception(data: pd.DataFrame) -> pd.DataFrame:
        mask = data[(data['sales'] == 0) & (data['pred'] == 0) & (data['plan'] == 0)].index
        data.loc[mask, 'acc_pred'] = 0
        data.loc[mask, 'acc_plan'] = 0

        return data

    def calc_summary(self, data: pd.DataFrame, mega_filter: bool) -> None:
        # If only view results on mega brand
        if mega_filter:
            data = data[data['mega_yn'] == 'Y']

        data_renew = self.renew_data(data=data)

        # Count the class
        result = pd.DataFrame()
        grp_col = self.cust_batch_list + self.item_batch_list + ['mega_yn']
        for label in ['pred', 'plan']:
            temp = data_renew[data_renew['gubun'] == label].copy()
            temp = self.classify_accuracy(data=temp)
            temp = self.count_class(data=temp, grp_col=grp_col)
            temp = temp.reset_index()
            temp['gubun'] = label
            result = pd.concat([result, temp])

        # Add customer class column
        summary = self.map_cust_class(data=result)

        # Group by cust class and (forecast/plan)
        summary = summary.groupby(by=['cust_class', 'gubun']).sum()

        summary_sum = summary.reset_index() \
            .groupby('gubun') \
            .sum() \
            .reset_index() \
            .copy()
        summary_sum['cust_class'] = '0.전체'
        summary_sum = summary_sum.set_index(['cust_class', 'gubun'])
        summary = summary.append(summary_sum).sort_index()

        # Sum all of the class
        summary['tot_cnt'] = summary.groupby(by=['cust_class', 'gubun']).sum().sum(axis=1).copy()

        # Calculate the rate
        summary = summary.div(summary['tot_cnt'], axis=0).reset_index()
        summary_pivot = summary.pivot(
            index='gubun',
            columns='cust_class',
            values=['cover_cnt', 'less_cnt', 'over_cnt', 'zero_cnt']
        )
        if self.division == 'SELL_OUT':
            summary_pivot = pd.DataFrame(summary_pivot.loc['pred', :]).T

        # Save the result
        if self.exec_cfg['save_file_yn']:
            if mega_filter:
                prefix = '_summary_mega_y.csv'
            else:
                prefix = '_summary.csv'
            name = str(self.hrchy['lvl']['item']) + '_' + str(self.acc_classify_standard) + prefix
            path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division + '_' + name)
            summary_pivot.to_csv(path, encoding='cp949')

    def map_cust_class(self, data):
        cust_class = []
        if self.division == 'SELL_IN':
            cust_class = [self.sp1_hrchy_map[self.division][sp1c] for sp1c in data['sp1_c_cd'].values]
        elif self.division == 'SELL_OUT':
            cust_class = [self.sp1_hrchy_map[self.division][sp1c] for sp1c in data['cust_grp_cd'].values]
        data['cust_class'] = cust_class

        return data

    def calc_db(self, data: pd.DataFrame) -> None:
        data_concat = self.renew_data(data=data)

        # Count the class
        result = pd.DataFrame()
        grp_col = self.cust_batch_list + self.item_batch_list + ['mega_yn']
        for label in ['pred', 'plan']:
            temp = data_concat[data_concat['gubun'] == label].copy()
            temp = self.classify_accuracy(data=temp)
            temp = self.count_class(data=temp, grp_col=grp_col)
            temp = temp.reset_index()
            temp['gubun'] = label
            result = pd.concat([result, temp])

        # Add name information
        result = self.add_information(data=result)

        if self.exec_cfg['save_db_yn']:
            self.save_result_on_db(data=result)

    def renew_data(self, data: pd.DataFrame):
        data_concat = pd.DataFrame()
        for label in ['pred', 'plan']:
            mask_col = self.cust_batch_list + self.item_list + ['mega_yn', 'sales', 'acc_' + label]
            temp = data[mask_col].copy()
            temp = temp.rename(columns={'acc_' + label: 'acc'})
            temp['gubun'] = label
            data_concat = pd.concat([data_concat, temp])

        return data_concat

    def compare_for_db(self, data: pd.DataFrame, kind: str):
        # Calculate the accuracy
        data = self.calc_accuracy(data=data, dividend='sales', divisor=kind)

        # Classify the accuracy
        data = self.classify_accuracy(data=data)

        # Count the class
        grp_col = self.cust_batch_list + self.item_batch_list + ['mega_yn']
        data = self.count_class(data=data, grp_col=grp_col)

        data['gubun'] = kind

        # Reset index
        data = data.reset_index()

        return data

    def merge_sales_pred_plan(self, sales: pd.DataFrame, pred_plan: pd.DataFrame):
        # merge sales and pred-plan dataset
        merge_col_else = ['start_week_day', 'week', 'cust_grp_cd']
        merged = pd.merge(pred_plan, sales, on=self.item_list + merge_col_else, how='left')

        merged['sales'] = merged['sales'].fillna(0)

        return merged

    def init(self) -> None:
        self.set_date()         # Set date
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()        # Set the hierarchy
        self.set_info()
        self.make_dir()         # Make the directory

    def set_info(self) -> None:
        self.set_info_cal()
        self.set_info_item()
        self.set_info_cust()
        self.set_info_sales_matrix()

    def set_info_cal(self):
        # Load the calendar dataset
        self.cal_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())

    def set_info_item(self) -> None:
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

        self.item_mst = item_info

    def set_info_cust(self) -> None:
        cust_nm = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_nm_master())
        cust_nm['code'] = cust_nm['code'].astype(str)
        cust_map = defaultdict(lambda: defaultdict(dict))
        for cust_type, code, name in zip(cust_nm['type'], cust_nm['code'], cust_nm['name']):
            cust_map[cust_type][code] = name
        # cust_nm = cust_nm.set_index('code')['name'].to_dict()

        self.cust_map = cust_map

    def set_info_sales_matrix(self) -> None:
        # Load the sales matrix
        sales_matrix = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_matrix())
        sales_matrix = sales_matrix.rename(columns={'sku_cd': 'item_cd'})
        self.sales_matrix = sales_matrix

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

    def set_level(self, item_lvl: int) -> None:
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

    def load_data_batch(self) -> pd.DataFrame:
        # Load the plan dataset
        pred_plan = None
        info_plan = {'yymmdd': self.date_sales['compare']['from']}
        if self.division == 'SELL_IN':
            pred_plan = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_plan_sell_in(**info_plan))
        elif self.division == 'SELL_OUT':
            pred_plan = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_plan_sell_out(**info_plan))

        pred_plan = pred_plan.rename(columns={'planed': 'plan'})

        # Load the sales dataset
        if self.load_sales_option == 'recent':
            info_sales_compare = {
                'division_cd': self.division,
                'start_week_day': self.date_sales['compare']['from']
            }
            sales = None
            if self.division == 'SELL_IN':  # Sell-In Dataset
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))
            elif self.division == 'SELL_OUT':  # Sell-Out Dataset
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))

            # Drop sales column in pred & plan dataset
            pred_plan = pred_plan.drop(columns='sales')

            # Merge sales dataset
            pred_plan = self.merge_sales_pred_plan(sales=sales, pred_plan=pred_plan)

        return pred_plan

    def load_data_dev(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load sales dataset
        info_sales_compare = {
            'division_cd': self.division,
            'start_week_day': self.date_sales['compare']['from']
        }
        sales = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))

        # Load prediction dataset
        path = os.path.join(self.root_path, 'prediction', self.exec_kind, self.data_vrsn_cd,
                            self.division + '_' + self.data_vrsn_cd + '_C1-P3-'
                            + self.pred_csv_map['name'][self.hrchy['key'][:-1]])
        pred = pd.read_csv(path, encoding=self.pred_csv_map['encoding'][self.hrchy['key'][:-1]])

        # Change the column list to lowercase
        pred.columns = [col.lower() for col in pred.columns]
        pred = pred.rename(columns={'yymmdd': 'start_week_day', 'result_sales': 'pred'})

        # Change data type
        pred['cust_grp_cd'] = pred['cust_grp_cd'].astype(str)
        pred['start_week_day'] = pred['start_week_day'].astype(str)
        if 'item_cd' in pred.columns:
            pred['item_cd'] = pred['item_cd'].astype(str)

        return sales, pred

    @staticmethod
    def calc_accuracy(data, dividend: str, divisor: str, label=''):
        temp = data.copy()
        result = util.func_accuracy(data=data, dividend=dividend, divisor=divisor)
        temp['acc' + label] = result

        return temp

    def classify_accuracy(self, data: pd.DataFrame, label='') -> pd.DataFrame:
        condition = [
            data['acc' + label] == 1,
            data['sales'] == 0,
            data['acc' + label] < 1 - self.acc_classify_standard,
            data['acc' + label] > 1 + self.acc_classify_standard
        ]
        class_label = label + '_cnt'
        values = ['cover' + class_label, 'zero' + class_label, 'less' + class_label, 'over' + class_label]
        data['class' + label] = np.select(condlist=condition, choicelist=values, default=None)
        data['class' + label] = data['class' + label].fillna('cover' + class_label)

        return data

    def count_class(self, data: pd.DataFrame, grp_col: list, label=''):
        class_cnt = data.groupby(by=grp_col + ['class' + label])['item_cd'] \
            .count() \
            .astype(int)\
            .reset_index() \
            .rename(columns={'item_cd': 'class_cnt'})

        class_cnt = class_cnt.pivot(
            index=grp_col,
            columns=['class' + label],
            values='class_cnt'
        ).fillna(0)

        for class_kind in self.classify_kind:
            if class_kind + label + '_cnt' not in class_cnt.columns:
                class_cnt[class_kind + label + '_cnt'] = 0

        return class_cnt

    @staticmethod
    def aggregate_count(data: pd.DataFrame, grp_col: list):
        # Aggregate the count on file format
        level_cnt = data.groupby(by=grp_col)['item_cd'] \
            .count() \
            .rename('level_cnt')

        return level_cnt

    def add_information(self, data):
        # Add naming
        data['sp2_c_nm'] = [self.cust_map['SP2_C'].get(code, code) for code in data['sp2_c_cd'].values]
        data['sp2_nm'] = [self.cust_map['SP2'].get(code, code) for code in data['sp2_cd'].values]
        data['sp1_c_nm'] = [self.cust_map['SP1_C'].get(code, code) for code in data['sp1_c_cd'].values]
        data['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in data['cust_grp_cd'].values]

        # Add item names
        item_info = self.item_mst.copy()
        item_info = item_info[self.item_batch_list + [code[:-2] + 'nm' for code in self.item_batch_list]]\
            .drop_duplicates()\
            .copy()
        item_info = item_info.dropna(how='all')
        item_info = item_info.fillna('UNDEFINED')

        data = pd.merge(data, item_info, how='left', on=self.item_batch_list)

        return data

    def calcaute_count_rate(self, data: pd.DataFrame, label: str):
        # Calculate rates of count
        for classify_kind in self.classify_kind:
            rate_label = classify_kind + label
            data[rate_label + '_rate'] = np.round(data[rate_label + '_cnt'] / data['level_cnt'], 2)

        return data

    def save_result_on_db(self, data):
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
        data_db['gubun'] = data_db['gubun'].replace({'plan': '판매계획', 'pred': '수요예측'})

        # add calendar data
        calendar = self.cal_mst[['yymmdd', 'yymm', 'week']].copy().drop_duplicates()
        data_db = pd.merge(data_db, calendar, on='yymmdd')

        # Delete previous result
        del_info = {
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'yymmdd': self.date_cfg['date']['compare']['from']
        }

        self.io.delete_from_db(sql=self.sql_conf.del_pred_plan_acc(**del_info))

        # Save result on the DB
        self.io.insert_to_db(df=data_db, tb_name='M4S_O110630')