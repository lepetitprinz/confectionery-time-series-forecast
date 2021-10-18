import common.util as util

import pandas as pd
from copy import deepcopy


class DataPrep(object):
    drop_col_sale = ['division_cd', 'seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'create_date', 'week', 'cust_cd']
    str_type_col = ['sku_cd']
    sim_col = ['']
    lag_option = {'w1': 1, 'w2': 2}

    def __init__(self, division: str, hrchy_lvl: int, lag: str, common: dict, date: dict):
        self.division = division
        self.date_col = 'yymmdd'
        self.input_col = ['discount']
        self.target_col = common['target_col']
        self.col_agg_map = {'sum': ['qty'],
                            'avg': ['discount', 'gsr_sum', 'rhm_avg', 'temp_avg', 'temp_max', 'temp_min']}
        self.resample_rule = common['resample_rule']
        self.date_range = pd.date_range(start=date['date_from'],
                                        end=date['date_to'],
                                        freq=common['resample_rule'])
        self.lag = lag

        # Data Configuration
        self.hrchy_lvl = hrchy_lvl
        self.hrchy_list = common['hrchy_item'].split(',')[:hrchy_lvl]

        #
        self.exg_list = []

    def preprocess(self, sales: pd.DataFrame, exg: dict):
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # Drop columns
        sales = sales.drop(columns=self.drop_col_sale)

        # convert data type
        for col in self.str_type_col:
            sales[col] = sales[col].astype(str)

        sales[self.input_col] = sales[self.input_col].fillna(0)

        # ------------------------------- #
        # 2. Preprocess Exogenous dataset
        # ------------------------------- #
        self.exg_list = [exg.lower() for exg in list(exg['all']['idx_cd'].unique())]

        exg_all = util.prep_exg_all(data=exg['all'])
        exg_partial = util.prep_exg_partial(data=exg['partial'])

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Convert data type
        sales[self.date_col] = pd.to_datetime(sales[self.date_col], format='%Y%m%d')
        exg_all[self.date_col] = pd.to_datetime(exg_all[self.date_col], format='%Y%m%d')

        # Merge sales data & exogenous data
        # data = pd.merge(sales, exg_all, on=self.date_col, how='left')    # ToDo: Exception
        data = sales

        data = data.set_index(keys=self.date_col)

        data_group = util.group(data=data, hrchy=self.hrchy_list, hrchy_lvl=self.hrchy_lvl-1)

        # Resampling
        data_resample = util.hrchy_recursion(hrchy_lvl=self.hrchy_lvl-1,
                                             fn=self.resample,
                                             df=data_group)

        # Drop columns
        data_drop = util.hrchy_recursion(hrchy_lvl=self.hrchy_lvl-1,
                                         fn=self.drop_column,
                                         df=data_resample)

        data_rag = util.hrchy_recursion(hrchy_lvl=self.hrchy_lvl-1,
                                        fn=self.lagging,
                                        df=data_drop)

        return data_rag

    def resample(self, df: pd.DataFrame):
        # Split by aggregation method
        cols = set(df.columns)
        col_avg = list(cols.intersection(set(self.col_agg_map['avg'])))
        col_sum = list(cols.intersection(set(self.col_agg_map['sum'])))

        # Average method
        df_avg_resampled = pd.DataFrame()
        if col_avg:
            df_avg = df[col_avg]
            df_avg_resampled = df_avg.resample(rule=self.resample_rule).mean()
            df_avg_resampled = df_avg_resampled.fillna(value=0)

        # Sum method
        df_sum_resampled = pd.DataFrame()
        if col_sum:
            df_sum = df[col_sum]
            df_sum_resampled = df_sum.resample(rule=self.resample_rule).sum()
            df_sum_resampled = df_sum_resampled.fillna(value=0)

        # Concatenate aggregation
        df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        # if len(df_resampled.index) != len(self.date_range):
        #     idx_add = list(set(self.date_range) - set(df_resampled.index))
        #     data_add = np.zeros((len(idx_add), df_resampled.shape[1]))
        #     df_add = pd.DataFrame(data_add, index=idx_add, columns=df_resampled.columns)
        #     df_resampled = df_resampled.append(df_add)
        #     df_resampled = df_resampled.sort_index()

        cols = self.hrchy_list[:self.hrchy_lvl + 1]
        data_level = df[cols].iloc[0].to_dict()
        data_lvl = pd.DataFrame(data_level, index=df_resampled.index)
        df_resampled = pd.concat([df_resampled, data_lvl], axis=1)

        return df_resampled

    def lagging(self, data):
        """
        params: option: w1 / w2 / w1-w2
        """
        lagged = deepcopy(data[self.target_col])
        lagged = lagged.shift(periods=self.lag_option[self.lag])
        lagged = pd.DataFrame(lagged.values, columns=[self.target_col + '_lag'], index=lagged.index)
        result = pd.concat([data, lagged], axis=1)
        result = result.fillna(0)

        return result

    def drop_column(self, data):
        result = data.drop(columns=self.hrchy_list)

        return result
