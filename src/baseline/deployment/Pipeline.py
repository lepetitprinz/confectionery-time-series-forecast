import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict
from baseline.analysis.ResultReport import ResultReport

import os
import warnings
warnings.filterwarnings("ignore")


class Pipeline(object):
    def __init__(self, division: str, lvl_cfg: dict, exec_cfg: dict, step_cfg: dict, exec_rslt_cfg: dict):
        """
        :param division: Sales (SELL-IN / SELL-OUT)
        :param lvl_cfg: Data Level Configuration
            - cust_lvl: Customer Level (Customer Group - Customer)
            - item_lvl: Item Level (Biz/Line/Brand/Item/SKU)
        :param exec_cfg: Data I/O Configuration
            - save_step_yn: Save Each Step Object or not
            - save_db_yn: Save Result to DB or not
            - decompose_yn: Decompose Sales data or not
            - scaling_yn: Scale data or not
            - impute_yn: Impute data or not
            - rm_outlier_yn: Remove outlier or not
            - feature selection_yn : feature selection
        :param step_cfg: Execute Configuration
        """
        # I/O & Execution Configuration
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg
        self.exec_rslt_cfg = exec_rslt_cfg

        self.decompose_yn = exec_cfg['decompose_yn']

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
            'key': "C" + str(lvl_cfg['cust_lvl']) + '-' + "P" + str(lvl_cfg['item_lvl']) + '-',
            'lvl': {
                'cust': lvl_cfg['cust_lvl'],
                'item': lvl_cfg['item_lvl']
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),
                'item': self.common['hrchy_item'].split(',')
            },
            'apply': self.common['hrchy_cust'].split(',')[:lvl_cfg['cust_lvl']] +
                     self.common['hrchy_item'].split(',')[:lvl_cfg['item_lvl']]
        }

        # Path Configuration
        self.path = {
            'load': util.make_path_baseline(module='data', division=division, data_vrsn=self.data_vrsn_cd,
                                            hrchy_lvl='', step='load', extension='csv'),
            'cns': util.make_path_baseline(module='data', division=division, data_vrsn=self.data_vrsn_cd, hrchy_lvl='',
                                           step='cns', extension='csv'),
            'prep': util.make_path_baseline(module='result', division=division, data_vrsn=self.data_vrsn_cd,
                                            hrchy_lvl=self.hrchy['key'], step='prep', extension='pickle'),
            'train': util.make_path_baseline(module='result', division=division, data_vrsn=self.data_vrsn_cd,
                                             hrchy_lvl=self.hrchy['key'], step='train', extension='pickle'),
            'train_score_best': util.make_path_baseline(module='result', division=division, data_vrsn=self.data_vrsn_cd,
                                                        hrchy_lvl=self.hrchy['key'], step='train_score_best',
                                                        extension='pickle'),
            'pred': util.make_path_baseline(module='result', division=division,  data_vrsn=self.data_vrsn_cd,
                                            hrchy_lvl=self.hrchy['key'], step='pred', extension='pickle'),
            'pred_best': util.make_path_baseline(module='result', division=division,  data_vrsn=self.data_vrsn_cd,
                                                 hrchy_lvl=self.hrchy['key'], step='pred_best', extension='pickle')
        }

    def run(self):
        # ================================================================================================= #
        # 1. Load the dataset
        # ================================================================================================= #
        sales = None
        if self.step_cfg['cls_load']:
            print("Step 1: Load the dataset\n")
            if self.division == 'SELL_IN':
                # sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date))
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_test(**self.date))
            elif self.division == 'SELL_OUT':
                sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out(**self.date))

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=sales, file_path=self.path['load'], data_type='csv')

            print("Data load is finished\n")
        # ================================================================================================= #
        # 2. Check Consistency
        # ================================================================================================= #
        if self.step_cfg['cls_cns']:
            print("Step 2: Check Consistency \n")
            if not self.step_cfg['cls_load']:
                sales = self.io.load_object(file_path=self.path['load'], data_type='csv')

            # Load error information
            err_grp_map = self.io.get_dict_from_db(
                sql=self.sql_conf.sql_err_grp_map(),
                key='COMM_DTL_CD',
                val='ATTR01_VAL'
            )

            # Initiate consistency check class
            cns = ConsistencyCheck(division=self.division, common=self.common, hrchy=self.hrchy, date=self.date,
                                   err_grp_map=err_grp_map, save_yn=False)

            # Execute Consistency check
            sales = cns.check(df=sales)

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=sales, file_path=self.path['cns'], data_type='csv')

            print("Consistency check is finished\n")

        # ================================================================================================= #
        # 3.0 Load Dataset
        # ================================================================================================= #
        # Load dataset
        # Exogenous dataset
        exg = self.io.get_df_from_db(sql=SqlConfig.sql_exg_data(partial_yn='N'))

        # ================================================================================================= #
        # 3. Data Preprocessing
        # ================================================================================================= #
        data_prep = None
        exg_list = None
        if self.step_cfg['cls_prep']:
            print("Step 3: Data Preprocessing\n")
            if not self.step_cfg['cls_cns']:
                sales = self.io.load_object(file_path=self.path['cns'], data_type='csv')

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                division=self.division,
                common=self.common,
                hrchy=self.hrchy,
                exec_cfg=self.exec_cfg
            )
            # Preprocessing the dataset
            data_prep, exg_list, hrchy_cnt = preprocess.preprocess(data=sales, exg=exg)
            self.hrchy['cnt'] = hrchy_cnt

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(
                    data=(data_prep, exg_list, hrchy_cnt),
                    file_path=self.path['prep'],
                    data_type='binary'
                )

            print("Data preprocessing is finished\n")
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

        mst_info = {
            'cust_code': cust_code,
            'cust_grp': cust_grp,
            'item_mst': item_mst,
            'cal_mst': cal_mst,
            'model_mst': model_mst,
            'param_grid': param_grid
        }

        # ================================================================================================= #
        # 4. Training
        # ================================================================================================= #
        scores_best = None
        if self.step_cfg['cls_train']:
            print("Step 4: Train\n")
            if not self.step_cfg['cls_prep']:
                data_prep, exg_list, hrchy_cnt = self.io.load_object(file_path=self.path['prep'], data_type='binary')
                self.hrchy['cnt'] = hrchy_cnt
            # Initiate train class
            training = Train(
                division=self.division,
                mst_info=mst_info,
                data_vrsn_cd=self.data_vrsn_cd,
                exg_list=exg_list,
                hrchy=self.hrchy,
                common=self.common,
                exec_cfg=self.exec_cfg
            )

            if not self.exec_rslt_cfg['train']:
                # Train the models
                scores = training.train(df=data_prep)

                # Save Step result
                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(data=scores, file_path=self.path['train'], data_type='binary')
            else:
                scores = self.io.load_object(file_path=self.path['train'], data_type='binary')

            # Make score result
            # All scores
            scores_db, score_info = training.make_score_result(
                data=scores,
                hrchy_key=self.hrchy['key'],
                fn=training.score_to_df
            )
            # Best scores
            scores_best, score_best_info = training.make_score_result(
                data=scores,
                hrchy_key=self.hrchy['key'],
                fn=training.best_score_to_df
            )

            # Exception (Insert Error)
            scores_db.loc[:, 'item_nm'] = ''
            scores_best.loc[:, 'item_nm'] = ''

            # Save best scores
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=scores_best, file_path=self.path['train_score_best'], data_type='binary')

            if self.exec_cfg['save_db_yn']:
                # Save best of the training scores on the DB table
                print("Save training all scores on DB")
                table_nm = 'M4S_I110410'
                score_info['table_nm'] = table_nm
                self.io.delete_from_db(sql=self.sql_conf.del_score(**score_info))
                self.io.insert_to_db(df=scores_db, tb_name=table_nm)

                # Save best of the training scores on the DB table
                print("Save training best scores on DB")
                table_nm = 'M4S_O110610'
                score_best_info['table_nm'] = table_nm
                self.io.delete_from_db(sql=self.sql_conf.del_score(**score_best_info))
                self.io.insert_to_db(df=scores_best, tb_name='M4S_O110610')

            print("Training is finished\n")
        # ================================================================================================= #
        # 5. Forecast
        # ================================================================================================= #
        pred_best = None
        if self.step_cfg['cls_pred']:
            print("Step 5: Forecast\n")
            if not self.step_cfg['cls_prep']:
                data_prep, exg_list, hrchy_cnt = self.io.load_object(file_path=self.path['prep'], data_type='binary')
                self.hrchy['cnt'] = hrchy_cnt

            if not self.step_cfg['cls_train']:
                scores_best = self.io.load_object(file_path=self.path['train_score_best'], data_type='binary')

            # Initiate predict class
            predict = Predict(
                division=self.division,
                mst_info=mst_info, date=self.date,
                data_vrsn_cd=self.data_vrsn_cd,
                exg_list=exg_list,
                hrchy=self.hrchy,
                common=self.common
            )
            if not self.exec_rslt_cfg['predict']:
                # Forecast the model
                prediction = predict.forecast(df=data_prep)

                # Save Step result
                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(data=prediction, file_path=self.path['pred'], data_type='binary')
            else:
                # Load predicted object
                prediction = self.io.load_object(file_path=self.path['pred'], data_type='binary')

            # Make database format
            pred_all, pred_info = predict.make_db_format_pred_all(df=prediction, hrchy_key=self.hrchy['key'])
            pred_best = predict.make_db_format_pred_best(pred=pred_all, score=scores_best)

            # pred_all.loc[:, 'item_nm'] = ''
            # pred_best.loc[:, 'item_nm'] = ''

            if self.exec_cfg['save_step_yn']:
                pred_all.to_csv(
                    os.path.join('..', '..', 'pred', 'pred_all.csv'),
                    index=False, encoding='CP949'
                )
                pred_best.to_csv(
                    os.path.join('..', '..', 'pred', 'pred_best.csv'),
                    index=False, encoding='CP949'
                )
                self.io.save_object(data=pred_best, file_path=self.path['pred_best'], data_type='binary')

            # Save the forecast results on the db table
            if self.exec_cfg['save_db_yn']:
                # Save prediction of all algorithms
                print("Save all of prediction results on DB")
                table_pred_all = 'M4S_I110400'
                pred_info['table_nm'] = table_pred_all
                self.io.delete_from_db(sql=self.sql_conf.del_prediction(**pred_info))
                self.io.insert_to_db(df=pred_all, tb_name=table_pred_all)

                # Save prediction of best algorithm
                print("Save best of prediction results on DB")
                table_pred_best = 'M4S_O110600'
                pred_info['table_nm'] = table_pred_best
                self.io.delete_from_db(sql=self.sql_conf.del_prediction(**pred_info))
                self.io.insert_to_db(df=pred_best, tb_name=table_pred_best)

            print("Forecast is finished\n")
        # ================================================================================================= #
        # 6. Report result
        # ================================================================================================= #
        if self.step_cfg['cls_rpt']:
            print("Step 6: Report result\n")
            if not self.step_cfg['cls_pred']:
                pred_best = self.io.load_object(file_path=self.path['pred_best'], data_type='binary')

            # Load compare dataset
            sales_comp = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_temp(
               **{'date_from': self.common['pred_start_day'],
                  'date_to': self.common['pred_end_day']}))

            item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())
            cust_grp_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_grp_info())
            cal_mst = self.io.get_df_from_db(sql=SqlConfig.sql_calendar())

            report = ResultReport(
                common=self.common,
                division=self.division,
                data_vrsn=self.data_vrsn_cd,
                hrchy=self.hrchy,
                cust_grp_mst=cust_grp_mst,
                item_mst=item_mst,
                cal_mst=cal_mst
            )
            report.compare_result(sales=sales_comp, pred=pred_best)

            # Close DB session
            self.io.session.close()

            print("Report results is finished\n")
