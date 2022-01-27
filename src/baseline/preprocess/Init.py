import common.util as util
from operation.Cycle import Cycle


class Init(object):
    """
    Time series setting class
    """
    def __init__(self, data_cfg: dict, exec_cfg: dict, common: dict, division: str, path_root: str):
        """
        :param data_cfg: Data configuration
        :param exec_cfg: Execution configuration
        :param common: common information
        :param division: division code (SELL-IN/SELL-OUT)
        :path_root: root path for baseline forecast
        """
        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg
        self.common = common
        self.division = division
        self.path_root = path_root

        # Setting
        self.data_vrsn_cd = ''    # Data version
        self.middle_out: bool = True    # Apply middle-out or not
        self.date = {}     # Date information (History / Middle-out)
        self.hrchy = {}    # Data hierarchy for customer & item
        self.level = {}    # Data hierarchy level for customer & item
        self.path = {}     # Save & Load path

    def run(self, cust_lvl: int, item_lvl: int) -> None:
        self.set_date()    # Set date information
        self.set_data_version()    # Set data version
        self.set_level(cust_lvl=cust_lvl, item_lvl=item_lvl)    # Set data hierarchy level
        self.set_hrchy()    # Set data hierarchy
        self.set_path()     # Set Save & Load path

    def set_date(self) -> None:
        if self.exec_cfg['cycle']:  # Executing demand forecast on weekly basis
            cycle = Cycle(common=self.common, rule=self.data_cfg['cycle'])
            cycle.calc_period()
            self.date = {
                'history': {    # Sales history period for forecast fitting
                    'from': cycle.hist_period[0],
                    'to': cycle.hist_period[1]
                },
                'middle_out': {    # Sales history period for middle-out
                    'from': cycle.eval_period[0],
                    'to': cycle.eval_period[1]
                },
                'evaluation': {    # Sales history period for comparing sales versus prediction
                    'from': cycle.pred_period[0],
                    'to': cycle.pred_period[1]
                }
            }
        else:    # Executing demand forecast on customized date
            self.date = self.data_cfg['date']

    def set_data_version(self) -> None:
        # Data version : [sales history(start)]-[sales history(end)]
        self.data_vrsn_cd = self.date['history']['from'] + '-' + self.date['history']['to']

    def set_level(self, cust_lvl: int, item_lvl: int):
        level = {
            'cust_lvl': cust_lvl,    # Customer level
            'item_lvl': item_lvl,    # Item level
            'middle_out': self.middle_out    # Execute Middle-out or not (True/False)
        }
        self.level = level

    def set_hrchy(self):
        hrchy = {
            'cnt': 0,    # Data level counts
            # Hierarchy Key (Format: C#-P#)
            # ex) C1-P5 -> Customer Level: 1 / Item Level: 5
            'key': "C" + str(self.level['cust_lvl']) + '-' + "P" + str(self.level['item_lvl']) + '-',
            # Hierarchy Level (Customer/Item)
            'lvl': {
                'cust': self.level['cust_lvl'],    # Customer level
                'item': self.level['item_lvl'],    # Item level
                'total': self.level['cust_lvl'] + self.level['item_lvl']    # Total level
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),    # Customer Level code list
                'item': self.common['hrchy_item'].split(',')     # Item Level code list
            },
            # Total code list (Customer + Item code)
            'apply': self.common['hrchy_cust'].split(',')[:self.level['cust_lvl']] +
                     self.common['hrchy_item'].split(',')[:self.level['item_lvl']]
        }
        self.hrchy = hrchy

    def set_path(self):
        path = {
            # History sales (csv)
            'load': util.make_path_baseline(
                path=self.path_root, module='data', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl='', step='load', extension='csv'),
            # Consistency check result (csv)
            'cns': util.make_path_baseline(
                path=self.path_root, module='data', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl='', step='cns', extension='csv'),
            # Data preprocessing result (Binary)
            'prep': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='prep', extension='pickle'),
            # Training result of all (Binary)
            'train': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='train', extension='pickle'),
            # Training result of best (Binary)
            'train_score_best': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='train_score_best', extension='pickle'),
            # Prediction step
            'pred': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred', extension='pickle'),
            # Prediction result of all (Binary)
            'pred_all': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred_all', extension='pickle'),
            # Prediction result of best (Binary)
            'pred_best': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred_best', extension='pickle'),
            # Training result of all (csv)
            'score_all_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='score_all', extension='csv'),
            # Training result of best (csv)
            'score_best_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='score_best', extension='csv'),
            # Prediction result of all (csv)
            'pred_all_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_all', extension='csv'),
            # Prediction result of best (csv)
            'pred_best_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_best', extension='csv'),
            'middle_out': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out', extension='csv'),
            # Middle-out result of all (csv)
            'middle_out_all': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out_all', extension='csv'),
            # Middle-out result of best (csv)
            'middle_out_best': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out_best', extension='csv'),
            'middle_out_db': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out_db', extension='csv'),
            'report': util.make_path_baseline(
                path=self.path_root, module='report', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='report', extension='csv'),
            'decompose': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='decompose', extension='csv'),
            'decompose_db': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='decompose_db', extension='csv'),
        }
        self.path = path
