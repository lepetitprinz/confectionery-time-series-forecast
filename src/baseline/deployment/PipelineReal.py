import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.Init import Init
from baseline.preprocess.DataLoad import DataLoad
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict
from baseline.analysis.ResultSummary import ResultSummary
from baseline.middle_out.MiddleOut import MiddleOut

import warnings
warnings.filterwarnings("ignore")


class PipelineReal(object):
    def __init__(self, data_cfg: dict, exec_cfg: dict, step_cfg: dict, exec_rslt_cfg: dict, unit_cfg: dict):
        """
        :param data_cfg: Data Configuration
        :param exec_cfg: Data I/O Configuration
        :param step_cfg: Execute Configuration
        """
        self.item_lvl = 3
        # Test version code
        self.test_vrsn_cd = data_cfg['test_vrsn_cd']

        # I/O & Execution Configuration
        self.data_cfg = data_cfg
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg
        self.exec_rslt_cfg = exec_rslt_cfg
        self.unit_cfg = unit_cfg

        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.data_lvl = self.io.get_df_from_db(
            sql=SqlConfig.sql_data_level()
        )

        # Data Configuration
        self.data_vrsn_cd = ''
        self.division = data_cfg['division']
        self.hrchy = {}
        self.level = {}
        self.path = {}
        self.date = {    # Todo : Test Exception
            'history': {
                'from': '20201102',
                'to': '20211031'
            },
            'middle_out': {
                'from': '20210802',
                'to': '20211031'
            },
            'evaluation': {
                'from': '20211101',
                'to': '20220130'
            }
        }

    def run(self):
        # ================================================================================================= #
        # 1. Initiate basic setting
        # ================================================================================================= #
        init = Init(
            data_cfg=self.data_cfg,
            exec_cfg=self.exec_cfg,
            common=self.common,
            division=self.division
        )
        init.run(cust_lvl=1, item_lvl=self.item_lvl)    # Todo : Exception

        # Set initialized object
        self.data_vrsn_cd = self.date['history']['from'] + '-' + self.date['history']['to']
        self.level = init.level
        self.hrchy = init.hrchy
        self.path = init.path

        # ================================================================================================= #
        # 0. Check the data version
        # ================================================================================================= #
        data_vrsn_list = self.io.get_df_from_db(sql=self.sql_conf.sql_data_version())
        if self.data_vrsn_cd not in list(data_vrsn_list['data_vrsn_cd']):
            data_vrsn_db = util.make_data_version(data_version=self.data_vrsn_cd)
            self.io.insert_to_db(df=data_vrsn_db, tb_name='M4S_I110420')

        # ================================================================================================= #
        # 2. Load the dataset
        # ================================================================================================= #
        sales = None
        load = DataLoad(
            io=self.io,
            sql_conf=self.sql_conf,
            data_cfg=self.data_cfg,
            unit_cfg=self.unit_cfg,
            date=self.date,
            division=self.division,
            data_vrsn_cd=self.data_vrsn_cd
        )

        if self.step_cfg['cls_load']:
            print("Step 1: Load the dataset\n")
            # Check data version
            load.check_data_version()

            # Load sales dataset
            sales = load.load_sales()

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=sales, file_path=self.path['load'], data_type='csv')

            print("Data load is finished\n")

        # Load master dataset
        mst_info = load.load_mst()

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
            cns = ConsistencyCheck(
                data_vrsn_cd=self.data_vrsn_cd,
                division=self.division,
                common=self.common,
                hrchy=self.hrchy,
                mst_info=mst_info,
                exec_cfg=self.exec_cfg,
                err_grp_map=err_grp_map
            )

            # Execute Consistency check
            sales = cns.check(df=sales)

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=sales, file_path=self.path['cns'], data_type='csv')

            print("Consistency check is finished\n")

        # ================================================================================================= #
        # 3. Data Preprocessing
        # ================================================================================================= #
        data_prep = None
        exg_list = None
        # Exogenous dataset
        exg = load.load_exog()

        if self.step_cfg['cls_prep']:
            print("Step 3: Data Preprocessing\n")
            if not self.step_cfg['cls_cns']:
                sales = self.io.load_object(file_path=self.path['cns'], data_type='csv')

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                common=self.common,
                hrchy=self.hrchy,
                data_cfg=self.data_cfg,
                exec_cfg=self.exec_cfg
            )

            if self.exec_cfg['rm_not_exist_lvl_yn']:
                sales_recent = self.io.get_df_from_db(
                    sql=self.sql_conf.sql_sell_in_week_grp_test(
                        **{'date_from': self.date['evaulation']['from'],
                           'date_to': self.date['evaulation']['to']}))
                preprocess.sales_recent = sales_recent

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
                data_vrsn_cd=self.data_vrsn_cd,    # Data version code
                division=self.division,            # Division code
                hrchy=self.hrchy,                  # Hierarchy
                common=self.common,                # Common information
                mst_info=mst_info,                 # Master information
                exg_list=exg_list,                 # Exogenous variable list
                data_cfg=self.data_cfg,            # Data configuration
                exec_cfg=self.exec_cfg             # Execute configuration
            )

            if not self.exec_rslt_cfg['train']:
                # Train the models
                scores = training.train(df=data_prep)

                # Save Step result
                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(data=scores, file_path=self.path['train'], data_type='binary')
            else:
                scores = self.io.load_object(file_path=self.path['train'], data_type='binary')

            # Save best parameters
            if self.exec_cfg['grid_search_yn']:
                training.save_best_params(scores=scores)

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

            scores_db.to_csv(self.path['score_all_csv'], index=False, encoding='cp949')
            scores_best.to_csv(self.path['score_best_csv'], index=False, encoding='cp949')

            # Remove Special Character
            if 'item_nm' in list(scores_db.columns):
                scores_db = util.remove_special_character(data=scores_db, feature='item_nm')
                scores_best = util.remove_special_character(data=scores_best, feature='item_nm')

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
                common=self.common,
                data_cfg=self.data_cfg
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

            pred_all.to_csv(self.path['pred_all_csv'], index=False, encoding='CP949')
            pred_best.to_csv(self.path['pred_best_csv'], index=False, encoding='CP949')

            # Remove Special Character
            if 'item_nm' in list(pred_all.columns):
                pred_all = util.remove_special_character(data=pred_all, feature='item_nm')
                pred_best = util.remove_special_character(data=pred_best, feature='item_nm')

            if self.exec_cfg['save_step_yn']:
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
        # 6. Middle Out
        # ================================================================================================= #
        if self.step_cfg['clss_mdout']:
            # Load item master
            item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())

            md_out = MiddleOut(
                common=self.common,
                division=self.division,
                data_vrsn=self.data_vrsn_cd,
                test_vrsn=self.test_vrsn_cd,
                hrchy=self.hrchy,
                ratio_lvl=5,
                item_mst=item_mst
            )
            # Load compare dataset
            date_recent = {
                'from': self.date['middle_out']['from'],
                'to': self.date['middle_out']['to']
            }
            sales_recent = None
            # Sell-In Dataset
            if self.division == 'SELL_IN':
                if self.data_cfg['in_out'] == 'out':
                    sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_week_grp(**date_recent))
                    # sql = self.sql_conf.sql_sell_in_week_grp_test(**date_recent))
                elif self.data_cfg['in_out'] == 'in':
                    sales_recent = self.io.get_df_from_db(
                        sql=self.sql_conf.sql_sell_in_week_grp_test_inqty(**date_recent))
            # Sell-Out Dataset
            elif self.division == 'SELL_OUT':
                if self.data_cfg['cycle'] == 'w':  # Weekly Prediction
                    sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week_grp_test(**date_recent))
                    # sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week_grp_test(**date_recent))
                elif self.data_cfg['cycle'] == 'm':  # Monthly Prediction
                    sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_month_grp_test(**date_recent))

            if not self.step_cfg['cls_pred']:
                pred_best = self.io.load_object(file_path=self.path['pred_best'], data_type='binary')

            # Run middle-out
            if not self.exec_rslt_cfg['predict']:
                data_ratio = md_out.prep_ratio(data=sales_recent)
                data_split = md_out.prep_split(data=pred_best)
                middle_out = md_out.middle_out(data_split=data_split, data_ratio=data_ratio)
                middle_out_db = md_out.after_processing(data=middle_out)

                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(
                        data=middle_out, file_path=self.path['middle_out'], data_type='csv')
                    self.io.save_object(
                        data=middle_out_db, file_path=self.path['middle_out_db'], data_type='csv')
            else:
                middle_out_db = self.io.load_object(file_path=self.path['middle_out_db'], data_type='csv')
                middle_info = md_out.add_del_information()

                if self.exec_cfg['save_db_yn']:
                    print("Save middle-out results on DB")
                    self.io.delete_from_db(sql=self.sql_conf.del_prediction(**middle_info))
                    self.io.insert_to_db(df=middle_out_db, tb_name='M4S_O110600')

                    # Save prediction of best algorithm to recent prediction table
                    self.io.delete_from_db(sql=self.sql_conf.del_pred_recent(**({'division_cd': self.division})))
                    self.io.insert_to_db(df=middle_out_db, tb_name='M4S_O111600')

            print("Middle-out is finished\n")
        # ================================================================================================= #
        # 7. Report result
        # ================================================================================================= #
        if self.step_cfg['cls_rpt']:
            print("Step 6: Report result\n")
            test_vrsn_cd = self.test_vrsn_cd

            if self.level['middle_out']:
                pred_best = self.io.load_object(file_path=self.path['middle_out'], data_type='csv')
                test_vrsn_cd = test_vrsn_cd + '_MIDDLE_OUT'
            else:
                if not self.step_cfg['cls_pred']:
                    pred_best = self.io.load_object(file_path=self.path['pred_best'], data_type='binary')

            # Load compare dataset
            date_compare = {
                'from': self.date['evaluation']['from'],
                'to': self.date['evaluation']['to']
            }
            date_recent = {
                'from': self.date['middle_out']['from'],
                'to': self.date['middle_out']['to']
            }

            sales_comp = None
            sales_recent = None

            # Sell-In dataset
            if self.division == 'SELL_IN':
                if self.data_cfg['in_out'] == 'out':
                    sales_comp = self.io.get_df_from_db(
                        sql=self.sql_conf.sql_sell_in_week_grp(**date_compare)
                        # sql=self.sql_conf.sql_sell_in_week_grp_test(**date_compare)
                    )
                    sales_recent = self.io.get_df_from_db(
                        sql=self.sql_conf.sql_sell_in_week_grp(**date_recent)
                        # sql=self.sql_conf.sql_sell_in_week_grp_test(**date_recent)
                    )
                elif self.data_cfg['in_out'] == 'in':
                    sales_comp = self.io.get_df_from_db(
                        sql=self.sql_conf.sql_sell_in_week_grp_test_inqty(**date_compare)
                    )
                    sales_recent = self.io.get_df_from_db(
                        sql=self.sql_conf.sql_sell_in_week_grp_test_inqty(**date_recent)
                    )
            # Sell-Out dataset
            elif self.division == 'SELL_OUT':
                if self.data_cfg['cycle'] == 'w':  # Weekly Prediction
                    sales_comp = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week_grp_test(**date_compare))
                    sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week_grp_test(**date_recent))
                elif self.data_cfg['cycle'] == 'm':  # Monthly Prediction
                    sales_comp = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_month_grp_test(**date_compare))
                    sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_month_grp_test(**date_recent))

            # Load item master
            item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())

            report = ResultSummary(
                data_vrsn=self.data_vrsn_cd,
                division=self.division,
                common=self.common,
                date=self.date,
                test_vrsn=test_vrsn_cd,
                hrchy=self.hrchy,
                item_mst=item_mst,
                lvl_cfg=self.level
            )
            result = report.compare_result(sales_comp=sales_comp, sales_recent=sales_recent, pred=pred_best)
            result, result_info = report.make_db_format(data=result)

            # Remove Special Character
            if 'item_nm' in list(result.columns):
                result = util.remove_special_character(data=result, feature='item_nm')

            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=result, file_path=self.path['report'], data_type='csv')

            if self.exec_cfg['save_db_yn']:
                print("Save prediction results on DB")
                self.io.delete_from_db(sql=self.sql_conf.del_compare_result(**result_info))
                self.io.insert_to_db(df=result, tb_name='M4S_O110620')

            # Close DB session
            self.io.session.close()

            print("Report results is finished\n")
