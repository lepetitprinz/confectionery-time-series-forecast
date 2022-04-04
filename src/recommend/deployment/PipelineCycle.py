from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from recommend.preprocess.Init import Init
from recommend.preprocess.DataPrep import DataPrep
from recommend.feature_engineering.Profiling import Profiling
from recommend.feature_engineering.Rank import Rank


class PipelineCycle(object):
    def __init__(self, cfg: dict):
        # Class instance attribute
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Execute instance attribute
        self.cfg = cfg

        # Data instance attribute
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.date = {}    # Date
        self.data_vrsn_cd = {}    # Data version

        self.item_col = 'sku_cd'    # Item code
        self.meta_col = 'bom_cd'    # Material code
        self.rank_n = 30      # Filter top n rank
        self.filter_n = 10    # Filter top best rank
        self.tb_name_rank = 'M4S_O110300'    # Ranking result table

    def run(self):
        # ====================================================================== #
        # 1. initiate basic setting
        # ====================================================================== #
        init = Init(cfg=self.cfg, common=self.common)
        init.run()

        # Set initialized object
        self.date = init.date
        self.data_vrsn_cd = init.data_vrsn_cd

        # ====================================================================== #
        # 2. Load Data
        # ====================================================================== #
        item_profile = self.io.get_df_from_db(sql=self.sql_conf.sql_bom_mst())   #
        item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())     # Item Master
        data_vrsn_list = self.io.get_df_from_db(sql=self.sql_conf.sql_data_version())
        calendar = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())    # Calendar Master

        # ====================================================================== #
        # 3. Data Preprocessing
        # ====================================================================== #
        prep = DataPrep(
            common=self.common,
            data_vrsn_cd=self.data_vrsn_cd,
            data_vrsn_list=data_vrsn_list,
            item_mst=item_mst,
            cal=calendar
        )
        data_prep = prep.preprocess(data=item_profile)

        # ====================================================================== #
        # 4. Profiling
        # ====================================================================== #
        profile = Profiling(item_col=self.item_col, meta_col=self.meta_col)
        similarity = profile.profiling(data=data_prep)

        # ====================================================================== #
        # 5. Ranking
        # ====================================================================== #
        rank = Rank(
            item_indices=profile.item_indices,
            similarity=similarity,
            top_n=self.rank_n
        )

        results = []
        item_list = list(profile.item_indices.keys())
        for item_cd in item_list:
            rank_result = rank.ranking(item_cd=item_cd)
            results.append([item_cd, rank_result])

        # ====================================================================== #
        # 6. Line Filtering (Optional)
        # ====================================================================== #
        results = prep.filter_rerank_by_line(data=results, filter_n=self.filter_n)

        # ====================================================================== #
        # 6. Save result
        # ====================================================================== #
        if self.cfg['save_db_yn']:
            results_db = prep.add_db_info(data=results)
            info = {'project_cd': self.common['project_cd'], 'data_vrsn_cd': self.data_vrsn_cd}
            self.io.delete_from_db(sql=self.sql_conf.del_profile(**info))
            self.io.insert_to_db(df=results_db, tb_name=self.tb_name_rank)
