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
                 grid_search_yn: bool, save_step_yn=False, load_step_yn=False, save_db_yn=False):
        # Class Configuration
        self.io = DataIO()

        # Data Configuration
        self.division = division
        self.hrchy_lvl = hrchy_lvl
        self.lag = lag

        # Execution Configuration
        self.grid_search_yn = grid_search_yn

        # Save & Load Configuration
        self.save_obj_yn = save_step_yn
        self.load_obj_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def run(self):
        # ====================== #
        # 1. Load the dataset
        # ====================== #
        data_load = None
        if config.CLS_SIM_LOAD:
            data_load = DataLoad(io=self.io, division=self.division, hrchy_lvl=self.hrchy_lvl, lag=self.lag)
            data_load.init()
            data_load.load()

            # Save step result
            if self.save_obj_yn:
                file_path = util.make_path_simulation(module='simulation', division=self.division, step='load',
                                                      extension='csv')
                self.io.save_object(data=data_load, file_path=file_path, data_type='csv')

        if self.load_obj_yn:
            file_path = util.make_path_simulation(module='simulation', division=self.division, step='load',
                                                  extension='csv')
            data_load = self.io.load_object(file_path=file_path, data_type='csv')

        # ====================== #
        # 2. Data Preprocessing
        # ====================== #
        if config.CLS_SIM_PREP:
            print("Step 2: Data Preprocessing\n")
            # Initiate data preprocessing class
            preprocess = DataPrep(
                division=self.division,
                common=data_load.common,
                date=data_load.date,
                hrchy_lvl=self.hrchy_lvl,
                lag=self.lag
            )

            # Preprocessing the dataset
            data_preped = preprocess.preprocess(sales=data_load.sales, exg=data_load.exg)

            # Save step result
            if self.save_obj_yn:
                file_path = util.make_path_simulation(module='simulation', division=self.division, step='prep',
                                                      extension='pickle')
                self.io.save_object(data=data_preped, file_path=file_path, data_type='binary')

        if self.load_obj_yn:
            file_path = util.make_path_simulation(module='simulation', division=self.division, step='prep', extension='csv')
            data_preped = self.io.load_object(file_path=file_path, data_type='csv')

        # ====================== #
        # 3. Training
        # ====================== #
        if config.CLS_SIM_TRAIN:
            print("Step3: Training")

            # Initiate data preprocessing class
            train = Train(
                data_version=data_load.data_version,
                hrchy_lvl=data_load.hrchy_lvl,
                algorithms=data_load.algorithms,
                parameters=data_load.parameters,
                grid_search_yn=self.grid_search_yn,
                save_obj_yn=self.save_obj_yn
            )


