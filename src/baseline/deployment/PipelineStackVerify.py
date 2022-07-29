from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.Init import Init
from baseline.preprocess.DataLoad import DataLoad
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.TrainStack import Train
from baseline.model.PredictStack import Predict
from middle_out.bak.MiddleOutBak import MiddleOut


import gc
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


class Pipeline(object):
    """
    Baseline forecast pipeline
    """
    def __init__(
            self,
            step_cfg: dict,
            data_cfg: dict,
            exec_cfg: dict,
            path_root: str
    ):
        """
        :param step_cfg: Pipeline step configuration
        :param data_cfg: Data configuration
        :param exec_cfg: Execution configuration
        :param path_root: root path for baseline forecast
        """
        self.item_lvl = 3    # Brand Level (Fixed)
        self.method = 'stack'

        # SP1-C
        self.sp1c_list = ('103', '107', '108', '109', '110')   # EC / Global
        self.sp1c_sp1 = None
        self.cust_grp = None

        # I/O & Execution instance attribute
        self.exec_kind = 'verify'
        self.data_cfg = data_cfg
        self.step_cfg = step_cfg
        self.exec_cfg = exec_cfg
        self.path_root = path_root

        # Object instance attribute
        self.io = DataIO()    # Data In/Out class
        self.sql_conf = SqlConfig()    # DB Query class
        self.common: dict = self.io.get_dict_from_db(    # common information dictionary
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

        # Data instance attribute
        self.division = data_cfg['division']    # division (SELL-IN/SELL-OUT)
        self.data_vrsn_cd = ''    # Data version
        self.hrchy = {}    # Data hierarchy for customer & item
        self.level = {}    # Data hierarchy level for customer & item
        self.date = {}     # Date information (History / Middle-out)
        self.path = {}     # Save & Load path

    def run(self):
        # ================================================================================================= #
        # 1. Initialize time series setting
        # ================================================================================================= #
        init = Init(
            data_cfg=self.data_cfg,
            exec_cfg=self.exec_cfg,
            common=self.common,
            division=self.division,
            path_root=self.path_root,
            exec_kind=self.exec_kind,
            method=self.method
        )
        init.run(cust_lvl=1, item_lvl=self.item_lvl)

        # Set initialized object
        self.date = init.date
        self.data_vrsn_cd = init.data_vrsn_cd
        self.level = init.level
        self.hrchy = init.hrchy
        self.path = init.path

        # ================================================================================================= #
        # 2. Load the dataset
        # ================================================================================================= #
        sales = None
        load = DataLoad(
            io=self.io,
            sql_conf=self.sql_conf,
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

        # Garbage collection
        gc.collect()
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
                err_grp_map=err_grp_map,
                path_root=self.path_root
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
        # sp1c <-> sp1
        sp1c_sp1 = self.io.get_df_from_db(sql=self.sql_conf.sql_sp1_sp1c(self.sp1c_list))
        sp1c_sp1['cust_grp_cd'] = sp1c_sp1['cust_grp_cd'].astype(str)
        self.sp1c_sp1 = sp1c_sp1
        self.cust_grp = tuple(self.sp1c_sp1['cust_grp_cd'].to_list())

        data_prep = None
        exg_list = None
        sales_dist = None

        # Exogenous dataset
        exg = load.load_exog()    # Weather dataset

        if (self.division == 'SELL_OUT') & (self.exec_cfg['add_exog_dist_sales']):
            sales_dist = load.load_sales_dist()

        if self.step_cfg['cls_prep']:
            print("Step 3: Data Preprocessing\n")
            if not self.step_cfg['cls_cns']:
                sales = self.io.load_object(file_path=self.path['cns'], data_type='csv')

            # Initiate data preprocessing class
            preprocess = DataPrep(
                date=self.date,
                common=self.common,
                hrchy=self.hrchy,
                division=self.division,
                data_cfg=self.data_cfg,
                exec_cfg=self.exec_cfg
            )

            # Filter sp1c
            sales['cust_grp_cd'] = sales['cust_grp_cd'].astype(str)
            sales = pd.merge(sales, sp1c_sp1, how='inner', on='cust_grp_cd')
            sales = sales.drop(columns=['sp1c_cd'])

            # Preprocessing the dataset
            data_prep, exg_list, hrchy_cnt = preprocess.preprocess(data=sales, weather=exg, sales_dist=sales_dist)
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
        ml_data_map = None

        if self.step_cfg['cls_train']:
            print("Step 4: Train\n")
            if not self.step_cfg['cls_prep']:
                data_prep, exg_list, hrchy_cnt = self.io.load_object(file_path=self.path['prep'], data_type='binary')
                self.hrchy['cnt'] = hrchy_cnt

            # Initiate train class
            training = Train(
                date=self.date,
                data_vrsn_cd=self.data_vrsn_cd,    # Data version code
                division=self.division,            # Division code
                hrchy=self.hrchy,                  # Hierarchy
                common=self.common,                # Common information
                mst_info=mst_info,                 # Master information
                exg_list=exg_list,                 # Exogenous variable list
                data_cfg=self.data_cfg,            # Data configuration
                exec_cfg=self.exec_cfg,             # Execute configuration
                path_root=self.path_root
            )

            # Train the models
            scores = training.train(df=data_prep)

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=scores, file_path=self.path['train'], data_type='binary')

            # Save best parameters of time series
            if self.exec_cfg['grid_search_yn']:
                training.save_best_params_ts(scores=scores)

            # Save best parameters of machine learning
            if self.exec_cfg['stack_grid_search_yn']:
                training.save_best_params_stack(scores=scores)

            else:
                # Make machine learning data
                ml_data_map = training.make_ml_data_map(
                    data=scores,
                    fn=training.make_hrchy_data_dict
                )

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
                    fn=training.make_best_score_df
                )

                # Save best scores
                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(data=scores_best, file_path=self.path['train_score_best'], data_type='binary')
                    self.io.save_object(data=ml_data_map, file_path=self.path['ml_data_map'], data_type='binary')

                if self.exec_cfg['save_db_yn']:
                    # Save best of the training scores on the DB table
                    print("Save training all scores on DB")
                    table_nm = 'M4S_I110410'
                    score_info['table_nm'] = table_nm
                    score_info['cust_grp_cd'] = self.cust_grp
                    self.io.delete_from_db(sql=self.sql_conf.del_score(**score_info))
                    self.io.insert_to_db(df=scores_db, tb_name=table_nm)

                    # Save best of the training scores on the DB table
                    print("Save training best scores on DB")
                    table_nm = 'M4S_O110610'
                    score_best_info['table_nm'] = table_nm
                    score_best_info['cust_grp_cd'] = self.cust_grp
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
                # Load best scores
                scores_best = self.io.load_object(file_path=self.path['train_score_best'], data_type='binary')
                ml_data_map = self.io.load_object(file_path=self.path['ml_data_map'], data_type='binary')

            # Initiate predict class
            predict = Predict(
                io=self.io,
                division=self.division,
                mst_info=mst_info,
                date=self.date,
                data_vrsn_cd=self.data_vrsn_cd,
                exg_list=exg_list,
                hrchy=self.hrchy,
                common=self.common,
                path_root=self.path_root,
                data_cfg=self.data_cfg,
                exec_cfg=self.exec_cfg,
                ml_data_map=ml_data_map
            )

            # Forecast
            prediction = predict.forecast(df=data_prep)

            # Save Step result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=prediction, file_path=self.path['pred'], data_type='binary')

            # Make database format
            pred_all, pred_info = predict.make_db_format_pred_all(df=prediction, hrchy_key=self.hrchy['key'])
            pred_best = predict.make_db_format_pred_best(pred=pred_all, score=scores_best)

            pred_all.to_csv(self.path['pred_all_csv'], index=False, encoding='CP949')
            pred_best.to_csv(self.path['pred_best_csv'], index=False, encoding='CP949')

            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=pred_best, file_path=self.path['pred_best'], data_type='binary')

            # Save the forecast results on the db table
            if self.exec_cfg['save_db_yn']:
                # Save prediction of all algorithms
                print("Save all of prediction results on DB")
                table_pred_all = 'M4S_I110400'
                pred_info['table_nm'] = table_pred_all
                pred_info['cust_grp_cd'] = self.cust_grp
                self.io.delete_from_db(sql=self.sql_conf.del_pred_all(**pred_info))
                self.io.insert_to_db(df=pred_all, tb_name=table_pred_all)

                # Save prediction results of best algorithm
                print("Save best of prediction results on DB")
                table_pred_best = 'M4S_O110600'
                pred_info['table_nm'] = table_pred_best
                pred_info['cust_grp_cd'] = self.cust_grp
                self.io.delete_from_db(sql=self.sql_conf.del_pred_best(**pred_info))
                self.io.insert_to_db(df=pred_best, tb_name=table_pred_best)

            print("Forecast is finished\n")

        # ================================================================================================= #
        # 6. Middle Out
        # ================================================================================================= #
        if self.step_cfg['cls_mdout']:
            print("Step 6: Middle Out\n")
            # Load item master
            item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())

            md_out = MiddleOut(
                common=self.common,
                division=self.division,
                data_vrsn=self.data_vrsn_cd,
                hrchy=self.hrchy,
                ratio_lvl=5,
                item_mst=item_mst
            )
            # Load compare dataset
            date_recent = {
                'from': self.date['middle_out']['from'],
                'to': self.date['middle_out']['to']
            }

            # Load sales dataset
            sales_recent = None
            if self.division == 'SELL_IN':    # Sell-In Dataset
                sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_week_grp(**date_recent))
            elif self.division == 'SELL_OUT':    # Sell-Out Dataset
                sales_recent = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week_grp(**date_recent))

            if not self.step_cfg['cls_pred']:
                pred_best = self.io.load_object(file_path=self.path['pred_best'], data_type='binary')

            # Run middle-out
            middle_out_db, _ = md_out.run_middle_out(sales=sales_recent, pred=pred_best)

            if self.exec_cfg['save_step_yn']:
                self.io.save_object(
                    data=middle_out_db, file_path=self.path['middle_out_db'], data_type='csv')

            if self.exec_cfg['save_db_yn']:
                middle_info = md_out.add_del_information()
                middle_info['cust_grp_cd'] = self.cust_grp
                # Save middle-out prediction of best algorithm to prediction table
                print("Save middle-out results on all result table")
                self.io.delete_from_db(sql=self.sql_conf.del_pred_best(**middle_info))
                self.io.insert_to_db(df=middle_out_db, tb_name='M4S_O110600')

                # Save prediction of best algorithm to recent prediction table
                print("Save middle-out results on recent result table")
                self.io.delete_from_db(sql=self.sql_conf.del_pred_recent(
                    **{'division_cd': self.division, 'cust_grp_cd': self.cust_grp}
                ))
                self.io.insert_to_db(df=middle_out_db, tb_name='M4S_O111600')

            # Close DB session
            self.io.session.close()

            print("Middle-out is finished\n")
