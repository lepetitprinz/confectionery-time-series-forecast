import common.util as util

import pandas as pd
from copy import deepcopy


class DataPrep(object):
    drop_col = ['division_cd', 'seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'create_date', 'week']

    def __init__(self, division: str, hrchy: dict, lag: str, common: dict, date: dict, threshold: int,
                 exec_cfg: dict):
        self.exec_cfg = exec_cfg
        self.division = division
        self.date_col = common['date_col']
        self.input_col = ['discount']
        self.target_col = common['target_col']
        self.conv_to_str_type_col = ['cust_grp_cd', 'sku_cd', self.date_col]
        self.col_agg_map = {'sum': common['agg_sum'].split(','), 'avg': common['agg_avg'].split(',')}
        self.resample_rule = 'w'
        self.date_range = pd.date_range(
            start=date['history']['from'],
            end=date['history']['to'],
            freq=common['resample_rule'])
        self.lag = lag
        self.threshold = threshold

        # Data Configuration
        self.hrchy = hrchy

    def preprocess(self, sales: pd.DataFrame, exg=None):
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # Drop columns
        sales = sales.drop(columns=self.drop_col)

        # convert data type
        for col in self.conv_to_str_type_col:
            sales[col] = sales[col].astype(str)

        sales[self.input_col] = sales[self.input_col].fillna(0)

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Convert data type to datetime
        sales[self.date_col] = pd.to_datetime(sales[self.date_col], format='%Y%m%d')

        # Merge sales data & exogenous data
        data = sales

        data = data.set_index(keys=self.date_col)

        data_group, hrchy_cnt = util.group(
            data=data,
            hrchy=self.hrchy['list'],
            hrchy_lvl=self.hrchy['lvl']-1)

        # Resampling
        data_resample = util.hrchy_recursion_with_none(
            hrchy_lvl=self.hrchy['lvl']-1,
            fn=self.resample,
            df=data_group
        )

        hrchy_cnt_filtered = util.counting(
            hrchy_lvl=self.hrchy['lvl']-1,
            df=data_resample
        )

        # Drop columns
        data_drop = util.hrchy_recursion(
            hrchy_lvl=self.hrchy['lvl']-1,
            fn=self.drop_column,
            df=data_resample
        )

        # lag sales data
        data_rag = util.hrchy_recursion(hrchy_lvl=self.hrchy['lvl']-1,
                                        fn=self.lagging,
                                        df=data_drop)

        return data_rag, hrchy_cnt, hrchy_cnt_filtered

    def resample(self, df: pd.DataFrame):
        # resampling
        df_sum_resampled = self.resample_by_agg(df=df, agg='sum')
        df_avg_resampled = self.resample_by_agg(df=df, agg='avg')

        # Concatenate aggregation
        df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        if self.exec_cfg['filter_threshold_week_yn']:
            if len(df_resampled[df_resampled['qty'] != 0]) < self.threshold:
                return None

        cols = self.hrchy['list']
        data_level = df[cols].iloc[0].to_dict()
        data_lvl = pd.DataFrame(data_level, index=df_resampled.index)
        df_resampled = pd.concat([data_lvl, df_resampled], axis=1)

        return df_resampled

    def resample_by_agg(self, df: pd.DataFrame, agg: str):
        resampled = None
        col_agg = set(df.columns).intersection(set(self.col_agg_map[agg]))
        if len(col_agg):
            resampled = df[col_agg]
            if agg == 'sum':
                resampled = resampled.resample(rule=self.resample_rule).sum()
            elif agg == 'avg':
                resampled = resampled.resample(rule=self.resample_rule).mean()

            # fill NaN
            resampled = resampled.fillna(value=0)

        return resampled

    def lagging(self, data):
        """
        params: option: w1 / w2 / w1-w2
        """
        lagged = deepcopy(data[self.target_col])
        if self.lag == 'w1':
            lagged = lagged.shift(periods=1)
            lagged = pd.DataFrame(lagged.values, columns=[self.target_col + '_lag'], index=lagged.index)
        elif self.lag == 'w2':
            lag1 = lagged.shift(periods=1)
            lag2 = lagged.shift(periods=1)
            lagged = (lag1 + lag2) / 2

        result = pd.concat([data, lagged], axis=1)
        result = result.fillna(0)

        return result

    def drop_column(self, data):
        result = data.drop(columns=self.hrchy['list'])

        return result
