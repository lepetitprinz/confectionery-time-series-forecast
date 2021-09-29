import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict
from baseline.tune.Split_bak import Split_bak


class Pipeline(object):
    def __init__(self, division: str, cust_lvl: int, item_lvl: int,
                 save_step_yn=False, load_step_yn=False, save_db_yn=False):
        """
        :param division: Sales (SELL-IN / SELL-OUT)
        :param cust_lvl: Customer Data Level ()
        :param item_lvl: Product Data Level (Biz/Line/Brand/Item/SKU)
        :param save_step_yn: Save result of each step
        """
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')

        # Data Configuration
        self.division = division
        self.target_col = self.common['target_col']
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}

        # Data Level Configuration
        self.hrchy_key = "C" + str(cust_lvl) + '-' + "P" + str(item_lvl) + '-'
        self.hrchy_lvl = {'cust_lvl': cust_lvl, 'item_lvl': item_lvl}
        self.hrchy_cust = self.common['hrchy_cust'].split(',')
        self.hrchy_item = self.common['hrchy_item'].split(',')
        self.hrchy_dict = {'hrchy_cust': self.common['hrchy_cust'].split(','),
                           'hrchy_item': self.common['hrchy_item'].split(',')}
        self.hrchy_list = self.hrchy_cust[:cust_lvl] + self.hrchy_item[:item_lvl]

        # Save & Load Configuration
        self.save_steps_yn = save_step_yn
        self.load_step_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        sell = None
        if config.CLS_LOAD:
            print("Step 1: Load the dataset\n")
            if self.division == 'SELL_IN':
                sell = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
            elif self.division == 'SELL_OUT':
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
        checked = None
        err_grp_map = self.io.get_dict_from_db(sql=self.sql_conf.sql_err_grp_map(), key='COMM_DTL_CD', val='ATTR01_VAL')
        if config.CLS_CNS:
            print("Step 2: Check Consistency \n")
            cns = ConsistencyCheck(division=self.division, common=self.common, hrchy=self.hrchy_list, date=self.date,
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
        # Load dataset
        # Customer dataset
        cust = self.io.get_df_from_db(sql=SqlConfig.sql_cust_code())
        # Exogenous dataset
        exg = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data())
        exg_list = list(idx.lower() for idx in exg['idx_cd'].unique())

        data_preped = None
        if config.CLS_PREP:
            print("Step 3: Data Preprocessing\n")
            # Initiate data preprocessing class
            preprocess = DataPrep(date=self.date, cust=cust, division=self.division,
                                  common=self.common, hrchy=self.hrchy_list)

            # Preprocess the dataset
            data_preped = preprocess.preprocess(data=checked, exg=exg)

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
        # 4.0. Load information
        # ================================================================================================= #
        # Load information form DB
        # Load master dataset
        cust_code = self.io.get_df_from_db(sql=SqlConfig.sql_cust_code())
        cust_grp = self.io.get_df_from_db(sql=SqlConfig.sql_cust_grp_info())
        item_mst = self.io.get_df_from_db(sql=SqlConfig.sql_item_view())
        cal_mst = self.io.get_df_from_db(sql=SqlConfig.sql_calendar())
        # item_mst['sku_cd'] = item_mst['sku_cd'].astype(str)

        # Load Algorithm & Hyper-parameter Information
        model_mst = self.io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'FCST'}))
        model_mst = model_mst.set_index(keys='model').to_dict('index')

        param_grid = self.io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())
        param_grid['stat_cd'] = param_grid['stat_cd'].apply(lambda x: x.lower())
        param_grid['option_cd'] = param_grid['option_cd'].apply(lambda x: x.lower())
        param_grid = util.make_lvl_key_val_map(df=param_grid, lvl='stat_cd', key='option_cd', val='option_val')

        mst_info = {'cust_code': cust_code,
                    'cust_grp': cust_grp,
                    'item_mst': item_mst,
                    'cal_mst': cal_mst,
                    'model_mst': model_mst,
                    'param_grid': param_grid}

        # ================================================================================================= #
        # 4. Training
        # ================================================================================================= #
        scores = None
        if config.CLS_TRAIN:
            print("Step 4: Train\n")
            # Initiate train class
            training = Train(division=self.division, mst_info=mst_info, date=self.date, exg_list=exg_list,
                             hrchy_lvl_dict=self.hrchy_lvl, hrchy_dict=self.hrchy_dict, common=self.common)

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
        data_pred = None
        if config.CLS_PRED:
            print("Step 5: Forecast\n")
            # Initiate predict class
            predict = Predict(division=self.division, mst_info=mst_info, date=self.date, exg_list=exg_list,
                              hrchy_lvl_dict=self.hrchy_lvl, hrchy_dict=self.hrchy_dict, common=self.common)

            # Forecast the model
            prediction = predict.forecast(df=data_preped)

            # Save Step result
            if self.save_steps_yn:
                file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                           step='pred', extension='pickle')
                self.io.save_object(data=prediction, file_path=file_path, data_type='binary')

            prediction_db = predict.make_pred_result(df=prediction, hrchy_key=self.hrchy_key)

            # Save the forecast results on the db table
            if self.save_db_yn:
                self.io.insert_to_db(df=prediction_db, tb_name='M4S_I110400')

            # Close DB session
            self.io.session.close()

        if self.load_step_yn:
            file_path = util.make_path(module='result', division=self.division, hrchy_lvl=self.hrchy_key,
                                       step='pred', extension='pickle')
            data_pred = self.io.load_object(file_path=file_path, data_type='binary')