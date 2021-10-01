import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from simulation.preprocess.DataPrep import DataPrep


class Pipeline(object):
    def __init__(self, division: str, hrchy_lvl: int, lag: str,
                 save_step_yn=False, load_step_yn=False, save_db_yn=False):
        # Class configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')

        # Data Configuration
        self.division = division
        self.target_col = self.common['target_col']
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}
        self.hrchy_lvl = hrchy_lvl
        self.lag = lag

        # Save & Load Configuration
        self.save_steps_yn = save_step_yn
        self.load_step_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def run(self):
        # ====================== #
        # 1. Load the dataset
        # ====================== #

        # 1.1 Sales dataset
        sales = None
        if config.CLS_WTIF_LOAD:
            # 1.1 Sales dataset
            print("Step 1: Load the dataset\n")
            if self.division == 'SELL_IN':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
            elif self.division == 'SELL_OUT':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))
            else:
                raise ValueError(f"{self.division} does not exist")

        if self.load_step_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl='',
                                       step='load', extension='csv')
            sales = self.io.load_object(file_path=file_path, data_type='csv')

        # 1.2 Exogenous dataset
        exg_all = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))
        exg_partial = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='Y'))
        exg_list = list(idx.lower() for idx in exg_all['idx_cd'].unique())
        exg = {'all': exg_all, 'partial': exg_partial}

        # ====================== #
        # 2. Data Preprocessing
        # ====================== #
        if config.CLS_WTIF_PREP:
            print("Step 2: Data Preprocessing\n")
            # Initiate data preprocessing class
            preprocess = DataPrep(division=self.division, common=self.common, date=self.date,
                                  hrchy_lvl=self.hrchy_lvl, lag=self.lag)

            # Preprocessing the dataset
            data_preped = preprocess.preprocess(sales=sales, exg=exg)

