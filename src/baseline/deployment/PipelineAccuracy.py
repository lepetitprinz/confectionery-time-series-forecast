from baseline.analysis.CalcAccuracy import CalcAccuracy

import os
import datetime


class PipelineAccuracy(object):
    def __init__(self, hist_to: str, division_list: list, item_lvl_list: list, exec_cfg: dict, opt_cfg: dict,
                 load_option: str):
        # Option
        self.exec_cfg = exec_cfg
        self.opt_cfg = opt_cfg
        self.date_cfg = {}

        # Data configuration
        self.load_option = load_option
        self.hist_to = hist_to
        self.division_list = division_list
        self.item_lvl_list = item_lvl_list
        self.date = {}

    def run(self):
        self.set_date()

        for division in self.division_list:
            for item_lvl in self.item_lvl_list:
                data_cfg = self.get_data_cfg(division=division, item_lvl=item_lvl)

                # Initiate class
                acc_cls = CalcAccuracy(
                    exec_cfg=self.exec_cfg,
                    opt_cfg=self.opt_cfg,
                    date_cfg=self.date_cfg,
                    data_cfg=data_cfg
                )
                acc_cls.run()

    def get_data_cfg(self, division: str, item_lvl: int):
        data_cfg = {
            'division': division,  # SELL_IN / SELL_OUT
            'item_lvl': item_lvl,
            'load_option': self.load_option,  # db / csv
        }

        return data_cfg

    def set_date(self):
        hist_to = self.hist_to

        # Change data type (string -> datetime)
        hist_to_datetime = datetime.datetime.strptime(hist_to, '%Y%m%d')

        # Add dates
        hist_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=156) + \
                    datetime.timedelta(days=1)
        compare_from = hist_to_datetime + datetime.timedelta(days=8)
        compare_to = hist_to_datetime + datetime.timedelta(days=14)

        # Change data type (datetime -> string)
        hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
        compare_from = datetime.datetime.strftime(compare_from, '%Y%m%d')
        compare_to = datetime.datetime.strftime(compare_to, '%Y%m%d')

        self.date_cfg = {
            'cycle_yn': False,
            'date': {
                'hist': {
                    'from': hist_from,
                    'to': hist_to
                },
                'compare': {
                    'from': compare_from,  # 20220110
                    'to': compare_to  # 20220116
                }
            },
            'data_vrsn_cd': hist_from + '-' + hist_to
        }