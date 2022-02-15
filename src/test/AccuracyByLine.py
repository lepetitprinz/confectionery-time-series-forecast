from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import datetime


class AccuracyByLine(object):
    div_line_sp1_map = {
        'SELL_IN': {
            'P1': ['1022', '1005', '1107', '1051', '1063', '1128'],
            'P2': ['1098', '1017', '1112', '1206', '1213', '1128', '1101']
        },
        'SELL_OUT': {
            'P1': ['1065', '1073']
        }
    }

    def __init__(self, monday: str):
        self.io = DataIO()
        self.sql_cfg = SqlConfig()
        self.data_version = ''
        self.start_monday_comp = ''
        self.start_monday_pred = monday

        # Options
        self.threshold = 5

    def run(self):
        self.set_data_version()
        self.calc_comp_start_monday()
        self.get_acc_by_line()

    def set_data_version(self):
        data_vrsn_df = self.io.get_df_from_db(sql=self.sql_cfg.sql_data_version())
        data_vrsn_df = data_vrsn_df[data_vrsn_df['exec_date'] == self.start_monday_pred]
        self.data_version = data_vrsn_df['data_vrsn_cd'].values[0]

    def calc_comp_start_monday(self) -> None:
        comp_monday = datetime.datetime.strptime(self.start_monday_pred, '%Y%m%d')
        comp_monday = comp_monday + datetime.timedelta(days=7)
        self.start_monday_comp = datetime.datetime.strftime(comp_monday, '%Y%m%d')

    def get_acc_by_line(self):
        for division, line in self.div_line_sp1_map.items():
            for line_code, sp1_list in line.items():
                for sp1 in sp1_list:
                    sql_info = {
                        'data_vrsn': self.data_version,
                        'yymmdd': self.start_monday_pred,
                        'yymmdd_comp': self.start_monday_comp,
                        'threshold': self.threshold,
                        'division': division,
                        'line': line_code,
                        'sp1': sp1
                    }
                    acc_by_line_df = self.io.get_df_from_db(sql=self.sql_cfg.sql_accuracy_by_line(**sql_info))
                    print("")

acc_by_line = AccuracyByLine(monday='20220131')
acc_by_line.run()