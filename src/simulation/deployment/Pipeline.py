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
        self.hrchy_lvl = hrchy_lvl
        self.lag = lag

        # Path Configuration
        self.path = {
            'load': util.make_path_sim(module='load', division=division, step='load', extension='csv'),

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
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))

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
                hrchy_lvl=self.hrchy_lvl,
                lag=self.lag,
            )

            # Preprocessing the dataset
            data_prep = preprocess.preprocess(sales=sales, exg=exg)

            # Save step result
            if self.exec_cfg['save_step_yn']:
                file_path = util.make_path_sim(module='simulation', division=self.division, step='prep',
                                               extension='pickle')
                self.io.save_object(data=data_prep, file_path=file_path, data_type='binary')

        else:
            file_path = util.make_path_sim(module='simulation', division=self.division, step='prep', extension='pickle')
            data_prep = self.io.load_object(file_path=file_path, data_type='binary')

        # ================================================================================================= #
        # 3. Training
        # ================================================================================================= #
        if self.step_cfg['cls_sim_train']:
            print("Step3: Training")
            # Load necessary dataset
            # Algorithm
            algorithms = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'SIM'}))
            parameters = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())

            # Initiate data preprocessing class
            train = Train(
                data_version=self.data_vrsn_cd,
                division=self.division,
                hrchy_lvl=self.hrchy_lvl,
                common=self.common,
                algorithms=algorithms,
                parameters=parameters,
                exec_cfg=self.exec_cfg
            )
            train.train(data=data_prep)
