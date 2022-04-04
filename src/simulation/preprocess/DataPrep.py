import common.util as util

import pandas as pd
from copy import deepcopy


class DataPrep(object):
    drop_col = ['division_cd', 'seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'create_date', 'week']

    def __init__(self, division: str, hrchy: dict, lag: str, common: dict, date: dict, threshold: int,
                 exec_cfg: dict):
        self.exec_cfg = exec_cfg    # Execution configuration
        self.division = division    # Division (SELL-IN/SELL-OUT)
        self.date_col = common['date_col']        # Date column name
        self.input_col = ['discount']             # Input column list
        self.target_col = common['target_col']    # Target column name
        self.conv_to_str_type_col = ['cust_grp_cd', 'sku_cd', self.date_col]
        self.col_agg_map = {'sum': common['agg_sum'].split(','), 'avg': common['agg_avg'].split(',')}
        self.resample_rule = 'w'
        self.date_range = pd.date_range(    # History date range (weekly)
            start=date['history']['from'],
            end=date['history']['to'],
            freq=common['resample_rule']
        )

        # Data instance attribute
        self.lag = lag    # lagging period
        self.hrchy = hrchy    # Hierarchy information
        self.threshold = threshold    # Minimum data length

    def preprocess(self, sales: pd.DataFrame, exg=None):
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # Drop unnecessary columns
        sales = sales.drop(columns=self.drop_col)

        # Change data type
        for col in self.conv_to_str_type_col:
            sales[col] = sales[col].astype(str)

        # Fill missing values with zero
        sales[self.input_col] = sales[self.input_col].fillna(0)

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Convert data type to datetime
        sales[self.date_col] = pd.to_datetime(sales[self.date_col], format='%Y%m%d')

        # Merge sales data & exogenous data
        data = sales

        # Set date column to index
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

        # Count filtered hierarchy level
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

        # Lag sales history data
        data_rag = util.hrchy_recursion(hrchy_lvl=self.hrchy['lvl']-1,
                                        fn=self.lagging,
                                        df=data_drop)

        return data_rag, hrchy_cnt, hrchy_cnt_filtered

    def resample(self, df: pd.DataFrame):
        # Resampling
        df_sum_resampled = self.resample_by_agg(df=df, agg='sum')
        df_avg_resampled = self.resample_by_agg(df=df, agg='avg')

        # Concatenate aggregation result
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
        col_agg = set(df.columns).intersection(set(self.col_agg_map[agg]))    # Get common columns
        if len(col_agg):
            resampled = df[col_agg]
            if agg == 'sum':    # Summation
                resampled = resampled.resample(rule=self.resample_rule).sum()
            elif agg == 'avg':    # Average
                resampled = resampled.resample(rule=self.resample_rule).mean()

            # Fill missing values with zero
            resampled = resampled.fillna(value=0)

        return resampled

    def lagging(self, data):
        """
        params: option: w1 / w2 / w1-w2
        """
        lagged = deepcopy(data[self.target_col])
        if self.lag == 'w1':
            lagged = lagged.shift(periods=1)    # Logging(1)
            lagged = pd.DataFrame(lagged.values, columns=[self.target_col + '_lag'], index=lagged.index)
        elif self.lag == 'w2':
            lag1 = lagged.shift(periods=1)    # Logging(1)
            lag2 = lagged.shift(periods=2)    # Logging(2)
            lagged = (lag1 + lag2) / 2    # Average lag 1 + lag 2

        result = pd.concat([data, lagged], axis=1)
        result = result.fillna(0)    # Fill missing values with zero

        return result

    def drop_column(self, data):
        result = data.drop(columns=self.hrchy['list'])

        return result
