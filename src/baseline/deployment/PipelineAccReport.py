from baseline.analysis.CalcAccuracyReport import CalcAccuracyReport

import datetime


class PipelineAccReport(object):
    item_lvl_map = {3: 'BRAND', 5: 'SKU'}

    def __init__(self, exec_kind: str, exec_cfg: dict, root_path: str, save_path: str,
                 division_list: list, item_lvl_list: list, hist_to='',
                 acc_classify_standard=0.25):
        # Execution instance attribute
        self.exec_kind = exec_kind
        self.exec_cfg = exec_cfg
        self.root_path = root_path
        self.save_path = save_path

        # Data instance attribute
        self.hist_to = hist_to
        self.division_list = division_list
        self.item_lvl_list = item_lvl_list
        self.date_cfg = {}
        self.acc_classify_standard = acc_classify_standard

        # Date instance attribute
        self.hist_range_week = 156
        self.this_monday_to_hist_to_range = 15
        self.comp_from_period = 8
        self.comp_to_period = 14

    def run(self):
        # Set necessary information for calculating accuracy
        self.init()

        for division in self.division_list:
            for item_lvl in self.item_lvl_list:
                data_cfg = self.get_data_cfg(division=division, item_lvl=item_lvl)

                # Initiate class
                acc = CalcAccuracyReport(
                    exec_kind=self.exec_kind,
                    exec_cfg=self.exec_cfg,
                    date_cfg=self.date_cfg,
                    data_cfg=data_cfg,
                    acc_classify_standard=self.acc_classify_standard
                )
                print("---------------------------------")
                print("Calculate Accuracy")
                print(f"Division: {division} / Level: {self.item_lvl_map[item_lvl]}")
                print("---------------------------------")

                # Calculate accuracy
                acc.run()

    def init(self):
        date = self.set_date()
        self.date_cfg = {
            'cycle_yn': self.exec_cfg['cycle_yn'],
            'date': {
                'hist': {
                    'from': date['hist_from'],
                    'to': date['hist_to']
                },
                'compare': {
                    'from': date['compare_from'],
                    'to': date['compare_to']
                }
            },
            'data_vrsn_cd': date['hist_from'] + '-' + date['hist_to']
        }

    def set_date(self):
        if self.exec_cfg['cycle_yn']:
            hist_to = self.calc_sunday_before_n_week()
        else:
            hist_to = self.hist_to

        hist_from, compare_from, compare_to = self.calc_date(hist_to=hist_to)

        # Change data type (datetime -> string)
        hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
        compare_from = datetime.datetime.strftime(compare_from, '%Y%m%d')
        compare_to = datetime.datetime.strftime(compare_to, '%Y%m%d')

        date = {
            'hist_from': hist_from,
            'hist_to': hist_to,
            'compare_from': compare_from,
            'compare_to': compare_to
        }

        return date

    def calc_sunday_before_n_week(self):
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        sunday_before_n_week = this_monday - datetime.timedelta(days=self.this_monday_to_hist_to_range)
        sunday_before_n_week = datetime.date.strftime(sunday_before_n_week, '%Y%m%d')

        return sunday_before_n_week

    def calc_date(self, hist_to):
        # Change data type (string -> datetime)
        hist_to = datetime.datetime.strptime(hist_to, '%Y%m%d')

        hist_from = hist_to - datetime.timedelta(weeks=self.hist_range_week) + datetime.timedelta(days=1)
        compare_from = hist_to + datetime.timedelta(days=self.comp_from_period)
        compare_to = hist_to + datetime.timedelta(days=self.comp_to_period)

        return hist_from, compare_from, compare_to

    def get_data_cfg(self, division: str, item_lvl: int):
        data_cfg = {
            'division': division,    # SELL_IN / SELL_OUT
            'item_lvl': item_lvl,
            'root_path': self.root_path,
            'save_path': self.save_path
        }

        return data_cfg
