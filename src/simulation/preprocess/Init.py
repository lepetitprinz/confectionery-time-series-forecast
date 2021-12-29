import common.util as util
from operation.Cycle import Cycle


class Init(object):
    def __init__(self, common: dict, division: str, path_root: str):
        self.division = division
        self.common = common
        self.path_root = path_root
        self.cycle = 'w'
        self.date = {}
        self.data_vrsn_cd = ''
        self.hrchy = {}
        self.path = {}

    def run(self):
        self.set_date()
        self.set_data_version()
        self.set_hrchy()
        self.set_path()

    def set_date(self) -> None:
        cycle = Cycle(common=self.common, rule=self.cycle)
        cycle.calc_period()
        self.date = {
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

    def set_data_version(self) -> None:
        self.data_vrsn_cd = self.date['history']['from'] + '-' + self.date['history']['to']

    def set_hrchy(self) -> None:
        self.hrchy = {
            'cnt_all': 0,
            'cnt_filtered': 0,
            'lvl': 6,
            'list': self.common['hrchy_cust'].split(',') + self.common['hrchy_item'].split(',')
        }

    def set_path(self) -> None:
        self.path = {
            'load': util.make_path_sim(
                path=self.path_root, module='load', division=self.division, data_vrsn=self.data_vrsn_cd,
                step='load', extension='csv'
            ),
            'prep': util.make_path_sim(
                path=self.path_root, module='prep', division=self.division, data_vrsn=self.data_vrsn_cd,
                step='prep', extension='pickle'
            )
        }
