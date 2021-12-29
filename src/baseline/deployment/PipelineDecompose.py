import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.Init import Init
from baseline.preprocess.DataPrep import DataPrep

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PipelineDecompose(object):
    def __init__(self, data_cfg: dict, exec_cfg: dict, exec_rslt_cfg: dict, item_lvl: int, path_root: str):
        """
        :param data_cfg: Data Configuration
        :param exec_cfg: Data I/O Configuration
        """
        self.cust_lvl = 0
        self.item_lvl = item_lvl

        # I/O & Execution Configuration
        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg
        self.exec_rslt_cfg = exec_rslt_cfg

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
        self.path_root = path_root
        self.division = 'SELL_IN'
        self.data_vrsn_cd = ''
        self.hrchy = {}
        self.level = {}
        self.path = {}
        self.date = {}

    def run(self):
        # ================================================================================================= #
        # 1. Initiate basic setting
        # ================================================================================================= #
        init = Init(
            path_root=self.path_root,
            data_cfg=self.data_cfg,
            exec_cfg=self.exec_cfg,
            common=self.common,
            division=self.division
        )
        init.run(cust_lvl=self.cust_lvl, item_lvl=self.item_lvl)    # Todo : Exception

        # Set initialized object
        self.date = init.date
        self.data_vrsn_cd = init.data_vrsn_cd
        self.level = init.level
        self.hrchy = init.hrchy
        self.path = init.path

        # ================================================================================================= #
        # 2. Time series decomposition
        # ================================================================================================= #
        print("Step 3: Time series decomposition\n")

        # Initiate data preprocessing class
        preprocess = DataPrep(
            date=self.date,
            common=self.common,
            hrchy=self.hrchy,
            data_cfg=self.data_cfg,
            exec_cfg=self.exec_cfg
        )

        # Preprocessing the dataset
        if not self.exec_rslt_cfg['decompose']:
            sales = self.io.load_object(file_path=self.path['cns'], data_type='csv')
            decomposed, exg_list, hrchy_cnt = preprocess.preprocess(data=sales, exg=pd.DataFrame())

            # Save the result
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=decomposed, file_path=self.path['decompose'], data_type='binary')

        else:
            decomposed = self.io.load_object(file_path=self.path['decompose'], data_type='binary')

        decomposed_list = util.hrchy_recursion_extend_key(
            hrchy_lvl=self.hrchy['lvl']['total']-1,
            df=decomposed,
            fn=preprocess.ravel_df
        )
        result = preprocess.conv_decomposed_list(data=decomposed_list)
        if self.exec_cfg['save_step_yn']:
            self.io.save_object(data=result, file_path=self.path['decompose_db'], data_type='csv')

        if self.exec_cfg['save_db_yn']:
            print("Save the result \n")
            info = {
                'project_cd': self.common['project_cd'],
                'data_vrsn_cd': self.data_vrsn_cd,
                'division_cd': self.division,
                'hrchy_lvl_cd': self.hrchy['key'][:-1]
            }
            self.io.delete_from_db(sql=self.sql_conf.del_decomposition(**info))
            self.io.insert_to_db(df=result, tb_name='M4S_O110500')

        print("Time series decomposition is finished\n")
