# Common class
import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

# Simulation Class
from simulation.preprocess.DataLoad import DataLoad
from simulation.preprocess.DataPrep import DataPrep
from simulation.model.Train import Train


class PipelineTest(object):
    def __init__(self, division: str, lag: str, step_cfg: dict, exec_cfg: dict):
        # I/O & Execution Configuration
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg

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
        self.data_vrsn_cd = self.common['rst_start_day'] + '-' + self.common['rst_end_day']
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}
        self.threshold = 10

        # Data Level Configuration
        self.hrchy = {
            'cnt_all': 0,
            'cnt_filtered': 0,
            'lvl': 6,
            'list': self.common['hrchy_cust'].split(',') + self.common['hrchy_item'].split(',')
        }
        self.lag = lag

        # Path Configuration
        self.path = {
            'load': util.make_path_sim(
                path=self.common['path_local'], module='load', division=division, data_vrsn=self.data_vrsn_cd,
                step='load', extension='csv'
            ),
            'prep': util.make_path_sim(
                path=self.common['path_local'], module='prep', division=division, data_vrsn=self.data_vrsn_cd,
                step='prep', extension='pickle'
            )
        }

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        sales = None
        if self.step_cfg['cls_sim_load']:
            if self.division == 'SELL_IN':
                # sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_test(**self.date))    # Todo: Temp data
            elif self.division == 'SELL_OUT':
                # sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**self.date))    # Todo: Temp data

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
            # exg = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                common=self.common,
                division=self.division,
                hrchy=self.hrchy,
                lag=self.lag,
                threshold=self.threshold,
                exec_cfg=self.exec_cfg
            )

            # Preprocessing the dataset
            data_prep, hrchy_cnt, hrchy_cnt_filtered = preprocess.preprocess(sales=sales)
            self.hrchy['cnt_all'] = hrchy_cnt
            self.hrchy['cnt_filtered'] = hrchy_cnt_filtered

            # Save step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(
                    data=(data_prep, hrchy_cnt, hrchy_cnt_filtered),
                    file_path=self.path['prep'],
                    data_type='binary'
                )

            print("Data preprocessing is finished.\n")
        # ================================================================================================= #
        # 3. Training
        # ================================================================================================= #
        if self.step_cfg['cls_sim_train']:
            print("Step3: Training")
            if not self.step_cfg['cls_sim_prep']:
                data_prep, hrchy_cnt, hrchy_cnt_filtered = self.io.load_object(
                    file_path=self.path['prep'],
                    data_type='binary'
                )
                self.hrchy['cnt_all'] = hrchy_cnt
                self.hrchy['cnt_filtered'] = hrchy_cnt_filtered

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
