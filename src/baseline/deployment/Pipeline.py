import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict


class Pipeline(object):
    def __init__(self, division: str, lvl_cfg: dict, io_cfg: dict, exec_cfg: dict):
        """
        :param division: Sales (SELL-IN / SELL-OUT)
        :param lvl_cfg: Data Level Configuration
            - cust_lvl: Customer Level (Customer Group - Customer)
            - item_lvl: Item Level (Biz/Line/Brand/Item/SKU)
        :param io_cfg: Data I/O Configuration
            - save_step_yn: Save Each Step Object or not
            - save_db_yn: Save Result to DB or not
            - decompose_yn: Decompose Sales data or not
        :param exec_cfg: Execute Configuration
        """
        # I/O & Execution Configuration
        self.exec_cfg = exec_cfg
        self.save_steps_yn = io_cfg['save_step_yn']
        self.save_db_yn = io_cfg['save_db_yn']
        self.decompose_yn = io_cfg['decompose_yn']

        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')

        # Data Configuration
        self.division = division
        self.target_col = self.common['target_col']
        self.date = {'date_from': self.common['rst_start_day'], 'date_to': self.common['rst_end_day']}
        self.data_vrsn_cd = self.date['date_from'] + '-' + self.date['date_to']
        # self.data_vrsn_cd = '20190915-20211003'  # Todo: Exception

        # Data Level Configuration
        self.hrchy_key = "C" + str(lvl_cfg['cust_lvl']) + '-' + "P" + str(lvl_cfg['item_lvl']) + '-'
        self.hrchy_lvl = {'cust_lvl': lvl_cfg['cust_lvl'], 'item_lvl': lvl_cfg['item_lvl']}
        self.hrchy_cust = self.common['hrchy_cust'].split(',')
        self.hrchy_item = self.common['hrchy_item'].split(',')
        self.hrchy_dict = {'hrchy_cust': self.common['hrchy_cust'].split(','),
                           'hrchy_item': self.common['hrchy_item'].split(',')}
        self.hrchy_list = self.hrchy_cust[:lvl_cfg['cust_lvl']] + self.hrchy_item[:lvl_cfg['item_lvl']]

        # Path Configuration
        self.path = {
            'load': util.make_path_baseline(module='data', division=division, hrchy_lvl='',
                                            step='load', extension='csv'),
            'cns': util.make_path_baseline(module='data', division=division, hrchy_lvl='',
                                           step='cns', extension='csv'),
            'prep': util.make_path_baseline(module='result', division=division, hrchy_lvl=self.hrchy_key,
                                            step='prep', extension='pickle'),
            'train': util.make_path_baseline(module='result', division=division, hrchy_lvl=self.hrchy_key,
                                             step='train', extension='pickle'),
            'pred': util.make_path_baseline(module='result', division=division, hrchy_lvl=self.hrchy_key,
                                            step='pred', extension='pickle')
        }

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        sales = None
        if self.exec_cfg['cls_load']:
            print("Step 1: Load the dataset\n")
            if self.division == 'SELL_IN':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
            elif self.division == 'SELL_OUT':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))

            # Save Step result
            if self.save_steps_yn:
                self.io.save_object(data=sales, file_path=self.path['load'], data_type='csv')
        # ================================================================================================= #
        # 2. Check Consistency
        # ================================================================================================= #
        checked = None
        if self.exec_cfg['cls_cns']:
            print("Step 2: Check Consistency \n")
            if not self.exec_cfg['cls_load']:
                sales = self.io.load_object(file_path=self.path['load'], data_type='csv')
            err_grp_map = self.io.get_dict_from_db(sql=self.sql_conf.sql_err_grp_map(), key='COMM_DTL_CD',
                                                   val='ATTR01_VAL')
            cns = ConsistencyCheck(division=self.division, common=self.common, hrchy=self.hrchy_list, date=self.date,
                                   err_grp_map=err_grp_map, save_yn=False)
            checked = cns.check(df=sales)

            # Save Step result
            if self.save_steps_yn:
                self.io.save_object(data=checked, file_path=self.path['cns'], data_type='csv')

        # ================================================================================================= #
        # 3.0 Load Dataset
        # ================================================================================================= #
        # Load dataset
        # Customer dataset
        cust = self.io.get_df_from_db(sql=SqlConfig.sql_cust_code())

        # Exogenous dataset
        exg_all = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))
        exg_partial = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='Y'))
        exg = {'all': exg_all, 'partial': exg_partial}
        exg_list = list(idx.lower() for idx in exg_all['idx_cd'].unique())

        # ================================================================================================= #
        # 3. Data Preprocessing
        # ================================================================================================= #
        data_preped = None
        if self.exec_cfg['cls_prep']:
            print("Step 3: Data Preprocessing\n")
            if not self.exec_cfg['cls_cns']:
                checked = self.io.load_object(file_path=self.path['cns'], data_type='csv')

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                cust=cust,
                division=self.division,
                common=self.common,
                hrchy=self.hrchy_list,
                decompose_yn=self.decompose_yn
            )
            # Temporary process
            # checked = preprocess.make_temp_data(df=checked)

            # Preprocessing the dataset
            data_preped = preprocess.preprocess(data=checked, exg=exg)

            # Save Step result
            if self.save_steps_yn:
                self.io.save_object(data=data_preped, file_path=self.path['prep'], data_type='binary')
        # ================================================================================================= #
        # 4.0. Load information
        # ================================================================================================= #
        # Load information form DB
        # Load master dataset
        cust_code = self.io.get_df_from_db(sql=SqlConfig.sql_cust_code())
        cust_grp = self.io.get_df_from_db(sql=SqlConfig.sql_cust_grp_info())
        item_mst = self.io.get_df_from_db(sql=SqlConfig.sql_item_view())
        cal_mst = self.io.get_df_from_db(sql=SqlConfig.sql_calendar())

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
        if self.exec_cfg['cls_train']:
            print("Step 4: Train\n")
            if not self.exec_cfg['cls_prep']:
                data_preped = self.io.load_object(file_path=self.path['prep'], data_type='binary')
            # Initiate train class
            training = Train(
                division=self.division,
                mst_info=mst_info,
                data_vrsn_cd=self.data_vrsn_cd,
                exg_list=exg_list,
                hrchy_lvl_dict=self.hrchy_lvl,
                hrchy_dict=self.hrchy_dict,
                common=self.common
            )
            # Train the models
            scores = training.train(df=data_preped)

            # Save Step result
            if self.save_steps_yn:
                self.io.save_object(data=scores, file_path=self.path['train'], data_type='binary')

            # Make score result
            # All scores
            scores_db, score_info = training.make_score_result(data=scores,
                                                               hrchy_key=self.hrchy_key,
                                                               fn=training.score_to_df)

            # Save all of the training scores on the DB table
            if self.save_db_yn:
                self.io.delete_from_db(sql=self.sql_conf.del_score(**score_info))
                self.io.insert_to_db(df=scores_db, tb_name='M4S_I110410')

            # Best scores
            scores_best_db, score_best_info = training.make_score_result(data=scores,
                                                                         hrchy_key=self.hrchy_key,
                                                                         fn=training.best_score_to_df)
            # Save best of the training scores on the DB table
            if self.save_db_yn:
                self.io.delete_from_db(sql=self.sql_conf.del_best_score(**score_best_info))
                self.io.insert_to_db(df=scores_best_db, tb_name='M4S_O110610')

        # ================================================================================================= #
        # 5. Forecast
        # ================================================================================================= #
        if self.exec_cfg['cls_pred']:
            print("Step 5: Forecast\n")
            if not self.exec_cfg['cls_prep']:
                data_preped = self.io.load_object(file_path=self.path['prep'], data_type='binary')

            # Initiate predict class
            predict = Predict(
                division=self.division,
                mst_info=mst_info, date=self.date,
                data_vrsn_cd=self.data_vrsn_cd,
                exg_list=exg_list,
                hrchy_lvl_dict=self.hrchy_lvl,
                hrchy_dict=self.hrchy_dict,
                common=self.common
            )
            # Forecast the model
            prediction = predict.forecast(df=data_preped)

            # Save Step result
            if self.save_steps_yn:
                self.io.save_object(data=prediction, file_path=self.path['pred'], data_type='binary')

            prediction_db, pred_info = predict.make_pred_result(df=prediction, hrchy_key=self.hrchy_key)

            # Save the forecast results on the db table
            if self.save_db_yn:
                self.io.delete_from_db(sql=self.sql_conf.del_prediction(**pred_info))
                self.io.insert_to_db(df=prediction_db, tb_name='M4S_I110400')

            # Close DB session
            self.io.session.close()
