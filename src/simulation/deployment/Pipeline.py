# Common class
import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

# Simulation Class
from simulation.preprocess.DataLoad import DataLoad
from simulation.preprocess.DataPrep import DataPrep
from simulation.model.Train import Train


class Pipeline(object):
    def __init__(self, division: str, hrchy_lvl: int, lag: str,
                 step_cfg: dict, exec_cfg: dict, exec_rslt_cfg: dict):
        # I/O & Execution Configuration
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg
        self.exec_rslt_cfg = exec_rslt_cfg

        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        # Data Configuration
        self.division = division
        self.date = {
            'date_from': self.common['rst_start_day'],
            'date_to': self.common['rst_end_day']
        }
        self.data_vrsn_cd = self.date['date_from'] + '-' + self.date['date_to']
        # Data Level Configuration
        self.hrchy = {
            'cnt': 0,
            'lvl': hrchy_lvl
        }
        self.lag = lag

        # Path Configuration
        self.path = {
            'load': util.make_path_sim(module='load', division=division, step='load', extension='csv'),
            'prep': util.make_path_sim(module='prep', division=division, step='prep', extension='pickle'),

        }

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        sales = None
        if self.step_cfg['cls_sim_load']:
            if self.division == 'SELL_IN':
                # sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_test(**self.date))  # Temp
            elif self.division == 'SELL_OUT':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**self.date))

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=sales, file_path=self.path['load'], data_type='csv')

        # ================================================================================================= #
        # 2. Data Preprocessing
        # ================================================================================================= #
        data_prep = None
        if self.step_cfg['cls_sim_prep']:
            print("Step 2: Data Preprocessing\n")
            if not self.step_cfg['cls_sim_load']:
                sales = self.io.load_object(file_path=self.path['load'], data_type='csv')

            # Load Exogenous dataset
            exg = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                common=self.common,
                division=self.division,
                hrchy=self.hrchy,
                lag=self.lag,
            )

            # Preprocessing the dataset
            data_prep, hrchy_cnt = preprocess.preprocess(sales=sales, exg=exg)
            self.hrchy['cnt'] = hrchy_cnt

            # Save step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=(data_prep, hrchy_cnt), file_path=self.path['prep'], data_type='binary')

            print("Data preprocessing is finished.\n")
        # ================================================================================================= #
        # 3. Training
        # ================================================================================================= #
        if self.step_cfg['cls_sim_train']:
            print("Step3: Training")
            if not self.step_cfg['cls_sim_prep']:
                data_prep, hrchy_cnt = self.io.load_object(file_path=self.path['prep'], data_type='binary')
                self.hrchy['cnt'] = hrchy_cnt

            # Load necessary dataset
            # Algorithm
            algorithms = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'SIM'}))
            best_params = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())

            # Initiate data preprocessing class
            train = Train(
                data_version=self.data_vrsn_cd,
                division=self.division,
                hrchy=self.hrchy,
                common=self.common,
                algorithms=algorithms,
                exec_cfg=self.exec_cfg
            )
            train.prep_params(best_params)
            train.train(data=data_prep)
