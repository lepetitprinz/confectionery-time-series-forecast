import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig


class DataLoad(object):
    def __init__(self, io, division: str, hrchy_lvl: int, lag: str):
        # initiate class
        self.io = io
        self.sql_conf = SqlConfig()

        # data option
        self.common = {}
        self.date = {}
        self.data_version = ''
        self.division = division
        self.hrchy_lvl = hrchy_lvl
        self.lag = lag

        # Dataset
        self.sales = {}
        self.exg = {}
        self.algorithms = {}
        self.parameters = {}

        # model configuration
        self.target_col = ''

    def init(self):
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}
        # self.data_version = str(self.date['date_from']) + '-' + str(self.date['date_to'])
        self.data_version = '20210101-20210530'
        self.target_col = self.common['target_col']

    def load(self):
        # Sales Dataset
        if self.division == 'sell_in':
            self.sales[self.division] = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
        elif self.division == 'sell_out':
            self.sales[self.division] = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))

        # Exogenous dataset
        exg_all = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))
        exg_partial = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='Y'))
        exg_list = list(idx.lower() for idx in exg_all['idx_cd'].unique())
        self.exg = {'all': exg_all, 'partial': exg_partial}

        # Algorithm
        algorithms = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'SIM'}))
        algorithms = algorithms.set_index(keys='model').to_dict('index')
        self.algorithms = algorithms

        # Hyper parameters
        param_grids = config.PARAM_GRIDS_SIM
        param_best = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())
        param_best['stat_cd'] = param_best['stat_cd'].apply(lambda x: x.lower())
        param_best['option_cd'] = param_best['option_cd'].apply(lambda x: x.lower())
        param_best = util.make_lvl_key_val_map(df=param_best, lvl='stat_cd', key='option_cd', val='option_val')
        self.parameters = {'param_grids': param_grids, 'param_best': param_best}
