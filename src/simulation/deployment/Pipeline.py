# Common class
import common.util as util
import common.config as config
from dao.DataIO import DataIO

# Simulation Class
from simulation.preprocess.DataLoad import DataLoad
from simulation.preprocess.DataPrep import DataPrep
from simulation.model.Train import Train


class Pipeline(object):
    def __init__(self, division: str, hrchy_lvl: int, lag: str,
                 grid_search_yn=False, save_obj_yn=False, load_obj_yn=False, save_db_yn=False):
        # Class Configuration
        self.io = DataIO()

        # Data Configuration
        self.division = division
        self.hrchy_lvl = hrchy_lvl
        self.lag = lag

        # Execution Configuration
        self.scaling_yn = True
        self.grid_search_yn = grid_search_yn

        # Save & Load Configuration
        self.save_obj_yn = save_obj_yn
        self.load_obj_yn = load_obj_yn
        self.save_db_yn = save_db_yn

    def run(self):
        # ====================== #
        # 1. Load the dataset
        # ====================== #
        data_load = None
        if config.CLS_SIM_LOAD:
            data_load = DataLoad(division=self.division, hrchy_lvl=self.hrchy_lvl, lag=self.lag,
                                 save_obj_yn=self.save_obj_yn, load_obj_yn=self.load_obj_yn)
            data_load.init(io=self.io)
            data_load.load(io=self.io)

        #     # Save step result
        #     if self.save_obj_yn:
        #         file_path = util.make_path_sim(
        #             module='simulation', division=self.division, step='load', extension='pickle')
        #         self.io.save_object(data=data_load, file_path=file_path, data_type='binary')
        #
        # if self.load_obj_yn:
        #     file_path = util.make_path_sim(
        #         module='simulation', division=self.division, step='load', extension='pickle')
        #     data_load = self.io.load_object(file_path=file_path, data_type='binary')

        # ====================== #
        # 2. Data Preprocessing
        # ====================== #
        data_prep = None
        if config.CLS_SIM_PREP:
            print("Step 2: Data Preprocessing\n")

            # Initiate data preprocessing class
            preprocess = DataPrep(
                division=self.division,
                hrchy_lvl=self.hrchy_lvl,
                lag=self.lag,
                common=data_load.common,
                date=data_load.date
            )

            # Preprocessing the dataset
            data_prep = preprocess.preprocess(
                sales=data_load.sales,
                exg=data_load.exg
            )

            # Save step result
            if self.save_obj_yn:
                file_path = util.make_path_sim(module='simulation', division=self.division, step='prep',
                                               extension='pickle')
                self.io.save_object(data=data_prep, file_path=file_path, data_type='binary')

        if self.load_obj_yn:
            file_path = util.make_path_sim(module='simulation', division=self.division, step='prep', extension='pickle')
            data_prep = self.io.load_object(file_path=file_path, data_type='binary')

        # ====================== #
        # 3. Training
        # ====================== #
        if config.CLS_SIM_TRAIN:
            print("Step3: Training")

            # Initiate data preprocessing class
            train = Train(
                data_version=data_load.data_version,
                hrchy_lvl=data_load.hrchy_lvl,
                common=data_load.common,
                algorithms=data_load.algorithms,
                parameters=data_load.parameters,
                scaling_yn=self.scaling_yn,
                grid_search_yn=self.grid_search_yn,
                save_obj_yn=self.save_obj_yn
            )
            train.train(data=data_prep)