from operation.Cycle import Cycle


class Init(object):
    def __init__(self, cfg: dict, common: dict):
        self.cfg = cfg
        self.common = common
        self.date = {}    # Date
        self.data_vrsn_cd = ''    # Data version

    def run(self):
        self.set_date()    # Set date
        self.set_data_version()    # Set data version

    def set_date(self):
        cycle = Cycle(common=self.common, rule=self.cfg['cycle'])
        cycle.calc_period()
        date = {
            'history': {
                'from': cycle.hist_period[0],    # History start date
                'to': cycle.hist_period[1]    # History end date
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

