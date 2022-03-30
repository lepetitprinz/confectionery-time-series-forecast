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


class CalcAccuracyReport(object):
    # all of item & date list
    item_list = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd']
    date_list = ['start_week_day', 'week']

    # Batch column list
    cust_batch_list = ['sp2_c_cd', 'sp2_cd', 'sp1_c_cd', 'cust_grp_cd']
    item_batch_list = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd']

    # Accuracy class
    classify_kind = ['cover', 'less', 'over', 'zero']
    item_lvl_map = {3: 'BRAND', 5: 'SKU'}

    summary_tag = {
        'all': '_summary.csv',
        'mega': '_summary_mega.csv'
    }

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

        # Filter data (Option)
        pred_plan = self.filter_data(data=pred_plan)

        self.calc_accuracy_pred_plan(data=pred_plan)

    def exec_dev(self) -> None:
        # Load the dataset
        sales_plan, pred = self.load_data_dev()

        # Preprocess the dataset
        merged = self.preprocess_dev(sales_plan=sales_plan, pred=pred)

        # Update recent sales matrix
        pred_plan = pd.merge(merged, self.sales_matrix, how='inner', on=['cust_grp_cd', 'item_cd'])

        # Filter data (Option)
        pred_plan = self.filter_data(data=pred_plan)

        self.calc_accuracy_pred_plan(data=pred_plan)

    def calc_accuracy_pred_plan(self, data: dict):
        raw, db = [], []
        summary = defaultdict(list)
        for label in ['pred', 'plan']:
            # Calculate accuracy
            acc = self.calc_accuracy(data=data[label], dividend='sales', divisor=label, label='')
            # Format: Raw
            raw.append(self.calc_raw(data=acc, label=label))
            # Format: Summary
            summary['all'].append(self.calc_summary(data=acc, label=label, mega_filter=False))
            summary['mega'].append(self.calc_summary(data=acc, label=label, mega_filter=True))
            # Format: DB
            db.append(self.calc_db(data=acc, label=label))

        # Save raw result
        self.save_raw(data=raw)
        # Save summary result
        self.save_summary(data=summary)
        # Save DB result
        self.save_db(data=db)

    def calc_raw(self, data, label: str):
        data = self.add_name_info(data=data, label=label)
        data = self.reorder_col_raw(data=data, label=label)

        return data

    def save_raw(self, data: list) -> None:
        # Save the result
        data = pd.concat(data, axis=0)
        path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '_raw.csv')
        data.to_csv(path, index=False, encoding='cp949')

    def calc_summary(self, data: pd.DataFrame, label: str, mega_filter: bool):
        if mega_filter:
            data = data[data['mega_yn'] == 'Y'].copy()

        summary_label = self.classify_accuracy(data=data)
        summary_label = self.count_class(
            data=summary_label,
            grp_col=self.cust_batch_list + self.item_batch_list + ['mega_yn']
        )
        summary_label = summary_label.reset_index()
        summary_label = self.map_cust_class(data=summary_label)
        summary_label['gubun'] = label
        summary_label = summary_label.groupby(by=['cust_class', 'gubun']).sum()

        # Add total result
        summary_label_sum = summary_label.reset_index() \
            .groupby('gubun') \
            .sum() \
            .reset_index() \
            .copy()
        summary_label_sum['cust_class'] = '0.전체'
        summary_label_sum = summary_label_sum.set_index(['cust_class', 'gubun'])
        summary_label = summary_label.append(summary_label_sum).sort_index()
        # summary_label = summary_label.drop(columns=['zero_cnt'])

        # Sum all of the class
        summary_label['tot_cnt'] = summary_label.groupby(by=['cust_class', 'gubun']).sum().sum(axis=1).copy()

        # Calculate the rate
        summary_label_rate = summary_label.div(summary_label['tot_cnt'], axis=0).copy()
        summary_label_rate = summary_label_rate.drop(columns=['tot_cnt'])
        summary_label_rate.columns = [col + '_rate' for col in summary_label_rate.columns]

        # Concatenate count & rate result
        summary_label_result = None
        if self.exec_cfg['summary_add_cnt']:
            summary_label_result = pd.concat([summary_label, summary_label_rate], axis=1)
            summary_label_result = summary_label_result[
                ['tot_cnt', 'cover_cnt', 'cover_cnt_rate', 'less_cnt', 'less_cnt_rate',
                 'over_cnt', 'over_cnt_rate', 'zero_cnt', 'zero_cnt_rate']
            ]

        else:
            summary_label_rate = summary_label_rate.reset_index()
            summary_label_result = summary_label_rate.pivot(
                index=['gubun'],
                columns='cust_class',
                values=['cover_cnt_rate', 'less_cnt_rate', 'over_cnt_rate', 'zero_cnt_rate']
            )

        return summary_label_result

    def save_summary(self, data) -> None:
        for tag in ['all', 'mega']:
            result = data[tag]
            result = pd.concat(result, axis=0)
            name = str(self.hrchy['lvl']['item']) + '_' + str(self.acc_classify_standard) + self.summary_tag[tag]
            path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division + '_' + name)
            result.to_csv(path, encoding='cp949')

    def calc_db(self, data: pd.DataFrame, label: str):
        db_label = self.classify_accuracy(data=data)
        db_label = self.count_class(
            data=db_label,
            grp_col=self.cust_batch_list + self.item_batch_list + ['mega_yn']
        )
        db_label = db_label.reset_index()
        db_label['gubun'] = label

        return db_label

    def save_db(self, data: list):
        data = pd.concat(data, axis=0)
        data = self.add_information(data=data)
        if self.exec_cfg['save_db_yn']:
            self.save_result_on_db(data=data)

    def add_name_info(self, data, label: str):
        data_add = data.copy()

        # Add classification
        data_add['gubun'] = label

        data_add['sp1_c_nm'] = [self.cust_map['SP1_C'].get(code, code) for code in data_add['sp1_c_cd'].values]
        data_add['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in data_add['cust_grp_cd'].values]

        # Add item information
        item_mst = self.item_mst.copy()
        item_mst = item_mst[['item_cd', 'item_nm', 'item_attr01_nm', 'item_attr02_nm',
                             'item_attr03_nm', 'item_attr04_nm']]
        item_mst = item_mst.fillna('UNDEFINED')

        data_add = pd.merge(data_add, item_mst, how='left', on=['item_cd'])

        return data_add

    def reorder_col_raw(self, data, label: str):
        # Reorder columns
        col_cust = ['sp1_c_nm', 'cust_grp_nm']
        col_item = ['item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'item_attr04_nm',
                    'item_cd', 'item_nm', 'mega_yn']
        col_etc = ['sales', label, 'acc', 'gubun']
        data = data[col_cust + col_item + col_etc].copy()
        data = data.rename(columns={label: 'qty'})

        return data

    def calc_dev(self, data: pd.DataFrame) -> None:
        acc = self.calc_accuracy(data=data, dividend='sales', divisor='pred', label='')

        summary = self.calc_summary(data=acc, label='pred', mega_filter=False)

        # Save the result
        path = os.path.join(self.save_path, self.data_vrsn_cd, self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '_dev_summary.csv')
        summary.to_csv(path, index=False, encoding='cp949')

    def filter_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Fill empty brand code
        data['item_attr03_cd'] = data['item_attr03_cd'].fillna('UNDEFINED')

        # Filter business range (Fixed)
        temp = data[data['item_attr01_cd'] == 'P1'].copy()

        # Filtering condition 1
        mask = temp[(temp['sales'] == 0) & (temp['pred'] == 0) & (temp['plan'] == 0)].index
        temp = temp.loc[set(temp.index) - set(mask)]

        # Filtering condition 2
        more = temp[temp['sales'] >= 5]
        less = temp[temp['sales'] < 5]

        # split into prediction and planning
        add_pred = less[(less['sales'] == 0) & (less['pred'] == 0)].copy()
        add_plan = less[(less['sales'] == 0) & (less['plan'] == 0)].copy()

        #
        pred = more.append(add_pred)
        plan = more.append(add_plan)

        #
        pred = pred.drop(columns=['plan'])
        plan = plan.drop(columns=['pred'])

        data_kind = {'pred': pred, 'plan': plan}

        return data_kind

    def preprocess_dev(self, sales_plan, pred) -> pd.DataFrame:
        sales_plan = sales_plan.drop(columns=['pred'])
        sales_plan = sales_plan.rename(columns={'planed': 'plan'})

        # Filter columns
        pred = pred[pred['start_week_day'] == self.date_sales['compare']['from']]
        comm_col = ['cust_grp_cd'] + self.item_list + self.date_list
        pred = pred[comm_col + ['pred']]

        # Merge dataset
        merged = pd.merge(sales_plan, pred, how='left', on=comm_col)
        merged = merged.fillna(0)

        return merged

    def map_cust_class(self, data):
        cust_class = []
        if self.division == 'SELL_IN':
            cust_class = [self.sp1_hrchy_map[self.division][sp1c] for sp1c in data['sp1_c_cd'].values]
        elif self.division == 'SELL_OUT':
            cust_class = [self.sp1_hrchy_map[self.division][sp1c] for sp1c in data['cust_grp_cd'].values]
        data['cust_class'] = cust_class

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
        sales_plan = None
        info_plan = {'yymmdd': self.date_sales['compare']['from']}
        if self.division == 'SELL_IN':
            sales_plan = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_plan_sell_in(**info_plan))
        elif self.division == 'SELL_OUT':
            sales_plan = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_plan_sell_out(**info_plan))

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

        return sales_plan, pred

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

    def add_information(self, data):
        # Add naming
        data['sp2_c_nm'] = [self.cust_map['SP2_C'].get(code, code) for code in data['sp2_c_cd'].values]
        data['sp2_nm'] = [self.cust_map['SP2'].get(code, code) for code in data['sp2_cd'].values]
        data['sp1_c_nm'] = [self.cust_map['SP1_C'].get(code, code) for code in data['sp1_c_cd'].values]
        data['cust_grp_nm'] = [self.cust_map['SP1'].get(code, code) for code in data['cust_grp_cd'].values]

        # Add item names
        item_info = self.item_mst[self.item_batch_list + [code[:-2] + 'nm' for code in self.item_batch_list]]\
            .drop_duplicates()\
            .copy()
        item_info = item_info.fillna('UNDEFINED')
        data = pd.merge(data, item_info, how='left', on=self.item_batch_list)

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
