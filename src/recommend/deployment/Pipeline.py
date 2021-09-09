from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from recommend.preprocess.DataPrep import DataPrep
from recommend.feature_engineering.Profiling import Profiling
from recommend.feature_engineering.Rank import Rank


class Pipeline(object):
    def __init__(self, item_col: str, meta_col: str):
        self.item_col = item_col
        self.meta_col = meta_col
        self.tb_name = 'M4S_O110300'
        self.sql_conf = SqlConfig()
        self.io = DataIO()

    def run(self):
        # ====================== #
        # 1. Load Data
        # ====================== #
        item_profile = self.io.get_df_from_db(sql=self.sql_conf.sql_item_profile())

        # ====================== #
        # 2. Data Preprocessing
        # ====================== #
        prep = DataPrep(item_col=self.item_col, meta_col=self.meta_col)
        data_prep = prep.preprocess(data=item_profile)

        # ====================== #
        # 3. Profiling
        # ====================== #
        profile = Profiling(item_col=self.item_col, meta_col=self.meta_col)
        similarity = profile.profiling(data=data_prep)

        # ====================== #
        # 4. Ranking
        # ====================== #
        rank = Rank(item_indices=profile.item_indices,
                    similarity=similarity,
                    top_n=10)

        results = []
        item_list = list(profile.item_indices.keys())
        for item_cd in item_list:
            rank_result = rank.ranking(item_cd=item_cd)
            results.append([item_cd, rank_result])

        # Save
        results_db = prep.conv_to_db_table(data=results)
        self.io.insert_to_db(df=results_db, tb_name=self.tb_name)
