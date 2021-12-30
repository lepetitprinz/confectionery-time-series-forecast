import common.util as util
from operation.Cycle import Cycle


class Init(object):
    def __init__(self, data_cfg: dict, exec_cfg: dict, common: dict, division: str, path_root: str):
        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg
        self.common = common
        self.division = division
        self.path_root = path_root

        # Setting
        self.data_vrsn_cd = ''
        self.middle_out = True
        self.date = {}
        self.hrchy = {}
        self.level = {}
        self.path = {}

    def run(self, cust_lvl: int, item_lvl: int):
        self.set_date()
        self.set_data_version()
        self.set_level(cust_lvl=cust_lvl, item_lvl=item_lvl)
        self.set_hrchy()
        self.set_path()

    def set_date(self):
        if self.exec_cfg['cycle']:
            cycle = Cycle(common=self.common, rule=self.data_cfg['cycle'])
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
        else:
            self.date = self.data_cfg['date']

    def set_data_version(self):
        self.data_vrsn_cd = self.date['history']['from'] + '-' + self.date['history']['to']

    def set_level(self, cust_lvl: int, item_lvl: int):
        level = {
            'cust_lvl': cust_lvl,    # Fixed
            'item_lvl': item_lvl,
            'middle_out': self.middle_out
        }
        self.level = level

    def set_hrchy(self):
        hrchy = {
            'cnt': 0,
            'key': "C" + str(self.level['cust_lvl']) + '-' + "P" + str(self.level['item_lvl']) + '-',
            'lvl': {
                'cust': self.level['cust_lvl'],
                'item': self.level['item_lvl'],
                'total': self.level['cust_lvl'] + self.level['item_lvl']
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),
                'item': self.common['hrchy_item'].split(',')
            },
            'apply': self.common['hrchy_cust'].split(',')[:self.level['cust_lvl']] +
                     self.common['hrchy_item'].split(',')[:self.level['item_lvl']]
        }
        self.hrchy = hrchy

    def set_path(self):
        path = {
            'load': util.make_path_baseline(
                path=self.path_root, module='data', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl='', step='load', extension='csv'),
            'cns': util.make_path_baseline(
                path=self.path_root, module='data', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl='', step='cns', extension='csv'),
            'prep': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='prep', extension='pickle'),
            'train': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='train', extension='pickle'),
            'train_score_best': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='train_score_best', extension='pickle'
            ),
            'pred': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred', extension='pickle'),
            'pred_all': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred_all', extension='pickle'),
            'pred_best': util.make_path_baseline(
                path=self.path_root, module='result', division=self.division, data_vrsn=self.data_vrsn_cd,
                hrchy_lvl=self.hrchy['key'], step='pred_best', extension='pickle'),
            'score_all_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='score_all', extension='csv'),
            'score_best_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='score_best', extension='csv'),
            'pred_all_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_all', extension='csv'),
            'pred_best_csv': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_best', extension='csv'),
            'middle_out': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out', extension='csv'),
            'middle_out_all': util.make_path_baseline(
                path=self.path_root, module='prediction', division=self.division,
                data_vrsn=self.data_vrsn_cd, hrchy_lvl=self.hrchy['key'], step='pred_middle_out_all', extension='csv'),
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
