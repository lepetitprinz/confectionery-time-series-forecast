import common.util as util
from operation.Cycle import Cycle


class Init(object):
    def __init__(self, common: dict, division: str, path_root: str):
        self.common = common          # Common information
        self.division = division      # Division (SELL-IN/SELL-OUT)
        self.path_root = path_root    # Root path
        self.cycle = 'w'              # Date resample standard: w(weekly)
        self.date = {}                # Date
        self.data_vrsn_cd = ''        # Data version
        self.hrchy = {}               # Hierarchy
        self.path = {}                # Path

    def run(self):
        self.set_date()    # Set date
        self.set_data_version()    # Set data version
        self.set_hrchy()    # Set Hierarchy
        self.set_path()    # Set path

    def set_date(self) -> None:
        cycle = Cycle(common=self.common, rule=self.cycle)
        cycle.calc_period()
        self.date = {
            'history': {
                'from': cycle.hist_period[0],    # History start day
                'to': cycle.hist_period[1]       # History end day
            },
            'middle_out': {
                'from': cycle.eval_period[0],    # Middle-out start day
                'to': cycle.eval_period[1]       # Middle-out end day
            },
            'evaluation': {
                'from': cycle.pred_period[0],    # Prediction start day
                'to': cycle.pred_period[1]       # Prediction end day
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
