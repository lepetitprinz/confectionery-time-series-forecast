# Common class
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

# Simulation Class
from simulation.preprocess.Init import Init
from simulation.preprocess.DataPrep import DataPrep
from simulation.model.Train import Train


class PipelineCycle(object):
    def __init__(self, step_cfg: dict, exec_cfg: dict, path_root: str):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # I/O & Execution Configuration
        self.step_cfg = step_cfg    # Step configuration
        self.exec_cfg = exec_cfg    # Execution configuration

        # Data configuration
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.path_root = path_root    # Root path
        self.division = 'SELL_IN'     # Division (SELL-IN/SELL-OUT)
        self.data_vrsn_cd = ''        # Data version
        self.date = {}                # Date
        self.path = {}                # Path

        # Data Level Configuration
        self.hrchy = {}        # Hierarchy
        self.lag = 'w1'        # Lag
        self.threshold = 20    # Minimum data length

    def run(self):
        # ================================================================================================= #
        # 1. Initiate basic setting
        # ================================================================================================= #
        init = Init(common=self.common, division=self.division, path_root=self.path_root)
        init.run()

        # Set initialized object
        self.date = init.date
        self.data_vrsn_cd = init.data_vrsn_cd
        self.hrchy = init.hrchy
        self.path = init.path

        # ================================================================================================= #
        # 2. Load the dataset
        # ================================================================================================= #
        sales = None
        if self.step_cfg['cls_sim_load']:
            print("Step 2: Load Dataset\n")
            date = {
                'from': self.date['history']['from'],
                'to': self.date['history']['to']
            }
            if self.division == 'SELL_IN':   # Load SELL-IN dataset
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**date))
            elif self.division == 'SELL_OUT':   # Load SELL-OUT dataset
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**date))

            # Save load step result
            # if self.exec_cfg['save_step_yn']:
            #     self.io.save_object(data=sales, file_path=self.path['load'], data_type='csv')

        # ================================================================================================= #
        # 3. Data Preprocessing
        # ================================================================================================= #
        data_prep = None
        if self.step_cfg['cls_sim_prep']:
            print("Step 3: Data Preprocessing\n")
            if not self.step_cfg['cls_sim_load']:
                sales = self.io.load_object(file_path=self.path['load'], data_type='csv')

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

            # Preprocess the dataset
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
                exec_cfg=self.exec_cfg,
                algorithms=algorithms,
                path_root=self.path_root
            )
            train.init(params=best_params)

            # Train what-if models
            train.train(data=data_prep)
