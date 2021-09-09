import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict


class Pipeline(object):
    def __init__(self, division: str, cust_lvl: int, prod_lvl: int,
                 save_step_yn=False, load_step_yn=False, save_db_yn=False):
        """
        :param division: Sales (SELL-IN / SELL-OUT)
        :param cust_lvl: Customer Data Level ()
        :param prod_lvl: Product Data Level (Biz/Line/Brand/Item/SKU)
        :param save_step_yn: Save result of each step
        """

        # Configuration
        # Data Configuration
        self.io = DataIO()    # Connect the DB
        self.sql_conf = SqlConfig()
        self.division = division
        self.hrchy_key = "C" + str(cust_lvl) + '-' + "P" + str(prod_lvl) + '-'
        self.hrchy = config.HRCHY_CUST[:cust_lvl] + config.HRCHY_PROD[:prod_lvl]

        # Date Configuration
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}

        # Save & Load Configuration
        self.save_steps_yn = save_step_yn
        self.load_step_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        print("Step 1: Load the dataset\n")
        sell = None
        if config.CLS_LOAD:
            if self.division == 'SELL-IN':
                sell = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
            elif self.division == 'SELL-OUT':
                sell = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))
            else:
                raise ValueError(f"{self.division} does not exist")

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='data', division=self.division, hrchy_lvl='',
                                           step='load', extension='csv')
                self.io.save_object(data=sell, file_path=file_path, data_type='csv')

        if self.load_step_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl='',
                                       step='load', extension='csv')
            sell = self.io.load_object(file_path=file_path, data_type='csv')

        # ================================================================================================= #
        # 2. Check Consistency
        # ================================================================================================= #
        print("Step 2: Check Consistency \n")
        checked = None
        err_grp_map = self.io.get_dict_from_db(sql=self.sql_conf.sql_err_grp_map(), key='COMM_DTL_CD', val='ATTR01_VAL')
        if config.CLS_CNS:
            cns = ConsistencyCheck(division=self.division,  hrchy=self.hrchy, date=self.date,
                                   err_grp_map=err_grp_map, save_yn=False)
            checked = cns.check(df=sell)

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='data', division=self.division, hrchy_lvl='',
                                           step='cns', extension='csv')
                self.io.save_object(data=checked, file_path=file_path, data_type='csv')

        if self.load_step_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl='',
                                       step='cns', extension='csv')
            checked = self.io.load_object(file_path=file_path, data_type='csv')

        # ================================================================================================= #
        # 3. Data Preprocessing
        # ================================================================================================= #
        print("Step 3: Data Preprocessing\n")
        data_preped = None
        if config.CLS_PREP:
            # Initiate data preprocessing class
            preprocess = DataPrep(date=self.date, division=self.division, hrchy=self.hrchy)

            # Preprocess the dataset
            data_preped = preprocess.preprocess(data=checked)

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                           step='prep', extension='pickle')
                self.io.save_object(data=data_preped, file_path=file_path, data_type='binary')

        if self.load_step_yn:
            file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                       step='prep', extension='pickle')
            data_preped = self.io.load_object(file_path=file_path, data_type='binary')

        # ================================================================================================= #
        # 4. Training
        # ================================================================================================= #
        # Load Algorithm & Hyper-parameter Information
        cand_models = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'FCST'}))
        model_info = cand_models.set_index(keys='model').to_dict('index')

        param_grid = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())
        param_grid['stat_cd'] = param_grid['stat_cd'].apply(lambda x: x.lower())
        param_grid['option_cd'] = param_grid['option_cd'].apply(lambda x: x.lower())
        param_grid = util.make_lvl_key_val_map(df=param_grid, lvl='stat_cd', key='option_cd', val='option_val')

        print("Step 4: Train\n")
        scores = None
        if config.CLS_TRAIN:
            # Initiate train class
            training = Train(division=self.division, model_info=model_info, param_grid=param_grid,
                             date=self.date, hrchy=self.hrchy)

            # Train the model
            scores = training.train(df=data_preped)

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                           step='train', extension='pickle')
                self.io.save_object(data=scores, file_path=file_path, data_type='binary')

            scores_db = training.make_score_result(data=scores, hrchy_key=self.hrchy_key)

            # Save the training scores on the DB table
            if self.save_db_yn:
                self.io.insert_to_db(df=scores_db, tb_name='M4S_I110410')

        if self.load_step_yn:
            file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                       step='prep', extension='pickle')
            data_preped = self.io.load_object(file_path=file_path, data_type='binary')

        # ================================================================================================= #
        # 5. Forecast
        # ================================================================================================= #
        print("Step 5: Forecast\n")
        if config.CLS_PRED:
            # Initiate predict class
            predict = Predict(division='SELL-IN', model_info=model_info, param_grid=param_grid,
                              date=self.date, hrchy=self.hrchy)

            # Forecast the model
            prediction = predict.forecast(df=data_preped)

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                           step='pred', extension='pickle')
                self.io.save_object(data=scores, file_path=file_path, data_type='binary')

            prediction_db = predict.make_pred_result(df=prediction, hrchy_key=self.hrchy_key)

            # Save the forecast results on the db table
            if self.save_db_yn:
                self.io.insert_to_db(df=prediction_db, tb_name='M4S_I110400')

            # Close DB session
            self.io.session.close()
