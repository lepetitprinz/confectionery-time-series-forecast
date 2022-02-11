import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class FeatureEngineering(object):
    def __init__(self, common, exg_list=None):
        self.common = common
        self.target_col = common['target_col']
        self.exg_list = exg_list
        self.decimal_point = 2

        # Feature Selection
        self.feat_select_method = 'spearmanr'    # pearson / spearmanr
        self.n_feature_to_select = 2

        # Time series rolling statistics
        self.rolling_window = 5
        self.rolling_center = True
        self.rolling_agg_list = ['min', 'max', 'mean', 'median']

        # Representative sampling
        self.repr_sampling_grp_list = ['sku_cd', 'yy', 'week']
        self.repr_sampling_agg_list = ['min', 'max', 'mean', 'median']

    # Feature selection
    def feature_selection(self, data: pd.DataFrame):
        drop_feat_list, exg_list = self.numeric_to_numeric(data=data)
        data = data.drop(columns=drop_feat_list)

        return data, exg_list

    def numeric_to_numeric(self, data):
        # feature selection with numeric to numeric
        target = data[self.target_col].values    # Target values

        coef_list = []
        for exg in self.exg_list:
            coef = 0
            if self.feat_select_method == 'spearmanr':    # Spearman correlation
                coef, p = spearmanr(target, data[exg].values)
            elif self.feat_select_method == 'pearson':    # Pearson correlation
                coef = np.corrcoef(target, data[exg].values)[0][1]
            coef_list.append((exg, abs(coef)))

        # Rank the feature importance
        coef_list = sorted(coef_list, key=lambda x: x[1], reverse=True)
        exg_list = coef_list[:self.n_feature_to_select]    # Selected variables & p-values
        drop_exg_list = coef_list[self.n_feature_to_select:]    # Removed variables  & p-values

        exg_list = [exg for exg, p in exg_list]    # Selected variables
        drop_exg_list = [exg for exg, p in drop_exg_list]    # Removed variables

        return drop_exg_list, exg_list

    # Representative sampling
    def repr_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        df['yymmdd'] = df.index
        df['yy'] = df['yymmdd'].dt.year

        repr_sample = df.groupby(by=self.repr_sampling_grp_list).agg(
            {self.target_col: self.repr_sampling_agg_list}
        ).round(self.decimal_point)

        # Reset indices
        repr_sample.columns = repr_sample.columns.get_level_values(1)
        repr_sample.columns = [self.target_col + '_' + col for col in repr_sample.columns]
        repr_sample = repr_sample.reset_index()

        # Merge dataset
        df = pd.merge(df, repr_sample, how='inner', on=self.repr_sampling_grp_list)

        df = df.drop(columns=['yy'])
        df = df.set_index('yymmdd')

        return df

    # Rolling Statistics Function
    def rolling(self, df: pd.DataFrame, feat: str) -> pd.DataFrame:
        df_roll = df.rolling(
            window=self.rolling_window,
            center=self.rolling_center,
            min_periods=1,
        ).agg({feat: self.rolling_agg_list}).round(self.decimal_point)

        df_roll.columns = df_roll.columns.get_level_values(1)
        df_roll.columns = [feat + '_' + col for col in df_roll.columns]

        df_roll = pd.merge(df, df_roll, how='inner', left_index=True, right_index=True)

        return df_roll
