import pandas as pd

from operation.Cycle import Cycle


class Init(object):
    def __init__(self, data_cfg: dict, common: dict):
        self.data_cfg = data_cfg
        self.common = common
        self.date = {}
        self.data_vrsn_cd = {}

    def run(self):
        self.set_date()

    def set_date(self):
        cycle = Cycle(common=self.common, rule=self.data_cfg['cycle'])
        cycle.calc_period()
        date = {
            'history': {
                'from': cycle.hist_period[0],
                'to': cycle.hist_period[1]
            },
            'middle_out': {
                'from': cycle.eval_period[0],
                'to': cycle.eval_period[1]
            },
            'evaluation': {
                'from': cycle.pred_period[0],
                'to': cycle.pred_period[1]
            }
        }

        self.date = date

    def set_data_version(self):
        self.data_vrsn_cd = self.date['history']['from'] + '-' + self.date['history']['to']