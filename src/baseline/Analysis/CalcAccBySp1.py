from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import os
import datetime
import pandas as pd


class CalcAccBySp1(object):
    grade_map = {'A': '75', 'B': '60', 'C': '50', 'F': 'ECC'}

    def __init__(self, biz_code, div_sp1_map: dict, root_path: str):
        # Object instance attribute
        self.io = DataIO()
        self.sql_cfg = SqlConfig()

        # Data instance attribute
        self.root_path = root_path
        self.save_path = os.path.join(root_path, 'analysis', 'accuracy', 'batch')
        self.data_version = ''
        self.start_monday_comp = ''
        self.biz_code = biz_code
        self.pred_exec_day = ''
        self.div_sp1_map = div_sp1_map

        # Option instance attribute
        self.threshold = 5
        self.pred_exec_range = 14

    def run(self):
        self.set_day()

        # Set the data version
        self.set_data_version()

        # Make accuracy table by each sp1
        self.make_acc_by_sp1()

        # Make accuracy table by each sp1 + line
        self.make_acc_by_line()

    def set_day(self):
        # Set prediction execution day
        self.set_pred_exec_day()

        # Set comparing start day
        self.set_comp_start_day()

    def set_pred_exec_day(self):
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        pred_exec_day = this_monday - datetime.timedelta(days=self.pred_exec_range)
        self.pred_exec_day = datetime.date.strftime(pred_exec_day, '%Y%m%d')

    def set_data_version(self):
        data_vrsn_df = self.io.get_df_from_db(sql=self.sql_cfg.sql_data_version())
        data_vrsn_df = data_vrsn_df[data_vrsn_df['exec_date'] == self.pred_exec_day]
        self.data_version = data_vrsn_df['data_vrsn_cd'].values[0]

    def set_comp_start_day(self) -> None:
        comp_monday = datetime.datetime.strptime(self.pred_exec_day, '%Y%m%d')
        comp_monday = comp_monday + datetime.timedelta(days=7)
        self.start_monday_comp = datetime.datetime.strftime(comp_monday, '%Y%m%d')

    def make_acc_by_sp1(self):
        temp = pd.DataFrame()
        for division, sp1_list in self.div_sp1_map.items():
            for sp1 in sp1_list:
                acc_by_sp1 = self.io.get_df_from_db(
                    sql=self.sql_cfg.sql_acc_by_sp1(
                        ** {
                            'division': division,
                            'data_vrsn': self.data_version,
                            'yymmdd': self.pred_exec_day,
                            'yymmdd_comp': self.start_monday_comp,
                            'threshold': self.threshold,
                            'biz': self.biz_code,
                            'sp1': sp1
                        }
                    ))
                acc_by_sp1 = self.change_sp1_sql_format(data=acc_by_sp1, division=division, sp1=sp1)
                temp = pd.concat([temp, acc_by_sp1], axis=0)

        temp_pivot = temp.pivot(index=['acc_grp'], columns=['division', 'sp1'], values='rate')
        temp_tot_cnt = temp.pivot(index=['acc_grp'], columns=['division', 'sp1'], values='total').iloc[0, :]
        temp_tot_cnt.name = 'tot_cnt'

        result = temp_pivot.append(temp_tot_cnt)
        self.save_result(data=result, kind='sp1')

    def save_result(self, data: pd.DataFrame, kind: str) -> None:
        # Define save path
        path = os.path.join(self.save_path, self.data_version)

        # If path dose not exist, then make the directory
        if not os.path.isdir(path):
            os.mkdir(path=path)

        # Save the result
        data.to_csv(os.path.join(path, self.data_version + '_' + self.biz_code + '_' + kind + '.csv'))

    def change_sp1_sql_format(self, data: pd.DataFrame, division: str, sp1: str):
        # Sum each count
        data['total'] = sum(data['count'])

        # Calculate rate of cumulative summation
        data['rate'] = data['cum_count'] / sum(data['count'])
        acc_by_sp1 = data[data['acc_grp'] != 'F']
        acc_by_sp1['acc_grp'] = acc_by_sp1['acc_grp'].apply(lambda x: self.grade_map[x])
        acc_by_sp1 = acc_by_sp1[['acc_grp', 'rate', 'total']]
        acc_by_sp1['division'] = division
        acc_by_sp1['sp1'] = sp1

        return acc_by_sp1

    def make_acc_by_line(self):
        temp = pd.DataFrame()
        for division, sp1_list in self.div_sp1_map.items():
            for sp1 in sp1_list:
                acc_by_line = self.io.get_df_from_db(
                    sql=self.sql_cfg.sql_acc_by_line(
                        **{
                            'division': division,
                            'data_vrsn': self.data_version,
                            'yymmdd': self.pred_exec_day,
                            'yymmdd_comp': self.start_monday_comp,
                            'threshold': self.threshold,
                            'biz': self.biz_code,
                            'sp1': sp1
                        }
                    ))
                acc_by_line = self.change_line_sql_format(data=acc_by_line, division=division, sp1=sp1)
                temp = pd.concat([temp, acc_by_line], axis=0)

        temp_pivot = temp.pivot(index=['item_attr02_cd'], columns=['division', 'sp1'], values='result')
        temp_pivot = temp_pivot.fillna(0.)

        self.save_result(data=temp_pivot, kind='line')

    @staticmethod
    def change_line_sql_format(data: pd.DataFrame, division: str, sp1: str):
        # Calculate rate of cumulative summation
        data['division'] = division
        data['sp1'] = sp1

        return data
