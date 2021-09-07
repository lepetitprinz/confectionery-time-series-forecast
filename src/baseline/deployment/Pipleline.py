import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict


class Pipeline(object):
    def __init__(self, division: str, prod_lvl: int):
        """
        :param division: Sales (SELL-IN / SELL-OUT)
        """
        # Data Configuration
        self.io = DataIO()
        self.division = division
        self.prod_lvl = prod_lvl
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')
        self.date = {'date_from': self.common['rst_start_day'],
                     'date_to': self.common['rst_end_day']}

        # Save Configuration
        self.save_steps_yn = True

    def run(self):
        # ---------------------- #
        # 1. Load the dataset
        # ---------------------- #
        print("Step 1: Load the dataset\n")
        sell = self.io.get_df_from_db(sql=SqlConfig.sql_sell_in(**self.date))

        # Save Step result
        if self.save_steps_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl=self.prod_lvl,
                                       step='load', data_type='csv')
            self.io.save_object(data=sell, file_path=file_path, kind='csv')

        # ---------------------- #
        # 2. Check Consistency
        # ---------------------- #
        print("Step 2: Check Consistency \n")
        cns = ConsistencyCheck(division=self.division, save_yn=False)
        checked = cns.check(df=sell)

        # Save Step result
        if self.save_steps_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl=self.prod_lvl,
                                       step='cns', data_type='csv')
            self.io.save_object(data=checked, file_path=file_path, kind='csv')

        # ---------------------- #
        # 3. Data Preprocessing
        # ---------------------- #
        print("Step 3: Data Preprocessing\n")
        preprocess = DataPrep(date=self.date)
        data_preped = preprocess.preprocess(data=checked, division=self.division)

        # Save Step result
        if self.save_steps_yn:
            file_path = util.make_path(module='data', division=self.division, hrchy_lvl=self.prod_lvl,
                                       step='prep', data_type='pickle')
            self.io.save_object(data=data_preped, file_path=file_path, kind='pickle')

        # ---------------------- #
        # 4. Training
        # ---------------------- #
        print("Step 4: Data Preprocessing\n")
        # Load Algorithm & Hyper-parameter Information
        cand_models = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'FCST'}))
        model_info = cand_models.set_index(keys='model').to_dict('index')

        param_grid = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())
        param_grid['stat'] = param_grid['stat'].apply(lambda x: x.lower())
        param_grid['option_cd'] = param_grid['option_cd'].apply(lambda x: x.lower())
        param_grid = util.make_lvl_key_val_map(df=param_grid, lvl='stat', key='option_cd', val='option_val')

        # Initiate train class
        training = Train(division=self.division, model_info=model_info, param_grid=param_grid, date=self.date)

        # Train the model
        scores = training.train(df=data_preped)

        # Save Step result
        if self.save_steps_yn:
            file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.prod_lvl,
                                       step='train', data_type='pickle')
            self.io.save_object(data=scores, file_path=file_path, kind='pickle')

        scores_db = training.make_score_result(data=scores)

        # Save the training scores on the DB table
        self.io.insert_to_db(df=scores_db, tb_name='M4S_I110410')

        # ---------------------- #
        # 5. Forecast
        # ---------------------- #
        print("Step 5: Forecast\n")
        # Initiate predict class
        predict = Predict(division='SELL-IN', model_info=model_info, param_grid=param_grid, date=self.date)

        # Forecast the model
        prediction = predict.forecast(df=data_preped)

        # Save Step result
        if self.save_steps_yn:
            file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.prod_lvl,
                                       step='pred', data_type='pickle')
            self.io.save_object(data=scores, file_path=file_path, kind='pickle')

        prediction_db = predict.make_pred_result(df=prediction)

        # Save the forecast results on the db table
        self.io.insert_to_db(df=prediction_db, tb_name='M4S_I110400')

        # Close DB session
        self.io.session.close()
