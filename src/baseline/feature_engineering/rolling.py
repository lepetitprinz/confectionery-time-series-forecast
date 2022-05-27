import pandas as pd


class FeatureEngineering(object):
    def __init__(self, common):
        self.common = common
        self.target_col = common['target_col']
        self.decimal_point = 2

        # Time series rolling statistics instance attribute
        self.rolling_window = 5
        self.rolling_center = True
        self.rolling_agg_list = ['min', 'max', 'mean', 'median']

        # Representative sampling instance attribute
        self.repr_sampling_grp_list = ['sku_cd', 'yy', 'week']
        self.repr_sampling_agg_list = ['min', 'max', 'mean', 'median']

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
