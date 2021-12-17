import common.util as util
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.Init import Init
from baseline.preprocess.DataPrep import DataPrep

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PipelineDecompose(object):
    def __init__(self, data_cfg: dict, exec_cfg: dict, item_lvl: int):
        """
        :param data_cfg: Data Configuration
        :param exec_cfg: Data I/O Configuration
        :param step_cfg: Execute Configuration
        """
        self.item_lvl = item_lvl

        # I/O & Execution Configuration
        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg

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
        self.division = 'SELL_IN'
        self.data_vrsn_cd = ''
        self.hrchy = {}
        self.level = {}
        self.path = {}
        self.date = {    # Todo : Test Exception
            'history': {
                'from': '20200706',
                'to': '20210704'
            },
            'middle_out': {
                'from': '20210405',
                'to': '20210704'
            },
            'evaluation': {
                'from': '20210705',
                'to': '20211003'
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
        init.run(cust_lvl=0, item_lvl=self.item_lvl)    # Todo : Exception

        # Set initialized object
        self.data_vrsn_cd = init.data_vrsn_cd
        self.level = init.level
        self.hrchy = init.hrchy
        self.path = init.path

        # ================================================================================================= #
        # 2. Time series decomposition
        # ================================================================================================= #
        print("Step 3: Time series decomposition\n")
        sales = self.io.load_object(file_path=self.path['cns'], data_type='csv')

        # Initiate data preprocessing class
        preprocess = DataPrep(
            date=self.date,
            common=self.common,
            hrchy=self.hrchy,
            data_cfg=self.data_cfg,
            exec_cfg=self.exec_cfg
        )

        # Preprocessing the dataset
        decomposed, exg_list, hrchy_cnt = preprocess.preprocess(data=sales, exg=pd.DataFrame())
        decomposed_list = util.hrchy_recursion_extend_key(
            hrchy_lvl=self.hrchy['lvl']['total']-1,
            df=decomposed,
            fn=preprocess.ravel_df
        )
        result = preprocess.conv_decomposed_list(data=decomposed_list)

        if self.exec_cfg['save_db_yn']:
            info = {
                'project_cd': self.common['project_cd'],
                'division_cd': self.division,
                'hrchy_lvl_cd': self.hrchy['key'][:-1]
            }
            self.io.delete_from_db(sql=self.sql_conf.del_decomposition(**info))
            self.io.insert_to_db(df=result, tb_name='M4S_O110500')

        print("Time series decomposition is finished\n")
