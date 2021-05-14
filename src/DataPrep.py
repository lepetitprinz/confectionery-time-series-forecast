import config

from collections import defaultdict
import pandas as pd
import numpy as np


class DataPrep(object):
    COL_DROP_SELL = ['pd_cd']
    COL_TYPE_NUM = ['amt', 'sales', 'unit_price', 'store_price']
    COL_TYPE_POS = ['amt', 'sales', 'unit_price', 'store_price']

    def __init__(self):
        # Path
        self.base_dir = config.BASE_DIR
        self.sell_in_dir = config.SELL_IN_DIR
        self.sell_out_dir = config.SELL_OUT_DIR

        # Condition
        self.variable_type = config.VAR_TYPE
        self.time_rule = config.RESAMPLE_RULE

        # Dataset
        self.sell_in_prep = None
        self.sell_out_prep = None
        self.prod_group = config.PROD_GROUP

        # Smoothing
        self.smooth_yn = config.SMOOTH_YN
        self.smooth_method = config.SMOOTH_METHOD
        self.smooth_rate = config.SMOOTH_RATE

        # run data preprocessing
        self.preprocess()

    def preprocess(self) -> None:
        print("Implement data preprocessing")
        # Sell in dataset

        # load dataset
        sell_in = pd.read_csv(self.sell_in_dir, delimiter='\t', thousands=',')
        sell_out = pd.read_csv(self.sell_out_dir, delimiter='\t', thousands=',')

        # preprocess sales dataset
        sell_in = self.prep_sales(df=sell_in)
        sell_out = self.prep_sales(df=sell_out)

        # convert target data type to float
        sell_in[config.COL_TARGET] = sell_in[config.COL_TARGET].astype(float)
        sell_out[config.COL_TARGET] = sell_out[config.COL_TARGET].astype(float)

        # Grouping
        sell_in_group = self.group(df=sell_in)
        sell_out_group = self.group(df=sell_out)

        # Univariate or Multivariate dataset
        sell_in_group = self.set_features(df_group=sell_in_group)
        sell_out_group = self.set_features(df_group=sell_out_group)

        # resampling
        resampled_sell_in = self.resample(df_group=sell_in_group)
        resampled_sell_out = self.resample(df_group=sell_out_group)

        self.sell_in_prep = resampled_sell_in
        self.sell_out_prep = resampled_sell_out

        print("Data preprocessing is finished\n")

    @ staticmethod
    def correct_target(df: pd.DataFrame) -> pd.DataFrame:
        df[config.COL_TARGET] = np.round(df['sales'] / df['store_price'])

        return df

    def prep_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert columns to lower case
        df.columns = [col.lower() for col in df.columns]

        # correct target or not
        if config.CRT_TARGET_YN:
            df = self.correct_target(df=df)

        # add product group column
        group_func = np.vectorize(self.group_product)
        df['prod_group'] = group_func(df['pd_nm'].to_numpy())

        # drop unnecessary columns
        df = df.drop(columns=self.__class__.COL_DROP_SELL)

        # convert date column to datetime
        df[config.COL_DATETIME] = pd.to_datetime(df[config.COL_DATETIME], format='%Y%m%d')

        # remove ',' from numbers and
        # for col in self.__class__.COL_TYPE_NUM:
        #     if col in df.columns and df[col].dtype != int:
        #         df[col] = df[col].str.replace(',', '')

        # fill NaN
        is_null_col = [col for col, is_null in zip(df.columns, df.isnull().sum()) if is_null > 0]
        for col in is_null_col:
            df[col] = df[col].fillna(0)

        # convert string type to int type
        for col in self.__class__.COL_TYPE_NUM:
            if col in df.columns and df[col].dtype != int:
                df[col] = df[col].astype(int)

        # Filter minus values from dataset
        for col in self.__class__.COL_TYPE_POS:
            if col in df.columns:
                df = df[df[col] >= 0].reset_index(drop=True)

        # add noise feature
        if config.ADD_EXO_YN:
            df = self.add_noise_feat(df=df)

        return df

    # Group mapping function
    def group_product(self, prod):
        return self.prod_group[prod]

    @staticmethod
    def group(df: pd.DataFrame) -> dict:
        df_group = defaultdict(lambda: defaultdict(dict))
        cust_list = list(df['cust_cd'].unique())
        if len(cust_list) > 1:    # if number of customers is more than 2
            cust_list.append('all')

        # Split customer groups
        for cust in cust_list:
            if cust != 'all':
                filtered_cust = df[df['cust_cd'] == cust]
            else:
                filtered_cust = df

            # Split product groups
            pd_group_list = list(filtered_cust['prod_group'].unique())
            if len(pd_group_list) > 1:    # if number of product groups is more than 2
                pd_group_list.append('all')
            for prod_group in pd_group_list:
                if prod_group == 'all':
                    filtered_prod_group = filtered_cust
                else:
                    filtered_prod_group = filtered_cust[filtered_cust['prod_group'] == prod_group]

                # Split products
                pd_list = list(filtered_prod_group['pd_nm'].unique())
                if len(pd_list) > 1:    # if number of products is more than 2
                    pd_list.append('all')
                for prod in pd_list:
                    if prod == 'all':
                        filtered_pd = filtered_prod_group
                    else:
                        filtered_pd = filtered_prod_group[filtered_prod_group['pd_nm'] == prod]
                    df_group[cust][prod_group].update({prod: filtered_pd})

        return df_group

    # @staticmethod
    # def group_bak(df: pd.DataFrame) -> dict:
    #     df_group = defaultdict(dict)
    #     cust_list = list(df['cust_cd'].unique())
    #     cust_list.append('all')
    #
    #     for cust in cust_list:
    #         if cust != 'all':
    #             filtered_cust = df[df['cust_cd'] == cust]
    #         else:
    #             filtered_cust = df
    #         pd_list = list(filtered_cust['pd_nm'].unique())
    #         pd_list.append('all')
    #         for prod in pd_list:
    #             if prod == 'all':
    #                 filtered_pd = filtered_cust
    #             else:
    #                 filtered_pd = filtered_cust[filtered_cust['pd_nm'] == prod]
    #             df_group[cust].update({prod: filtered_pd})
    #
    #     return df_group
    @staticmethod
    def add_noise_feat(df: pd.DataFrame) -> pd.DataFrame:
        vals = df[config.COL_TARGET].values * 0.05
        vals = vals.astype(int)
        vals = np.where(vals == 0, 1, vals)
        noise = np.random.randint(-vals, vals)
        df['exo'] = df[config.COL_TARGET].values + noise

        return df

    def set_features(self, df_group: dict) -> dict:
        for customers in df_group.values():
            for prod_group, products in customers.items():
                for product, df in products.items():
                    df = df[config.COL_TOTAL[self.variable_type]]
                    customers[prod_group][product] = df

        return df_group

    # def set_features_bak(self, df_group: dict) -> dict:
    #     for group in df_group.values():
    #         for key, val in group.items():
    #             val = val[config.COL_TOTAL[self.variable_type]]
    #             group[key] = val
    #
    #     return df_group

    def resample(self, df_group: dict) -> dict:
        """
        Data Resampling
        :param df_group: time series dataset
        :return:
        """
        for customers in df_group.values():
            for prod_group, products in customers.items():
                for product, df in products.items():
                    resampled = defaultdict(dict)
                    for rule in self.time_rule:
                        df_resampled = df.set_index(config.COL_DATETIME)
                        df_resampled = df_resampled.resample(rule=rule).sum()
                        if self.smooth_yn:
                            df_resampled = self.smoothing(df=df_resampled)
                        resampled[rule] = df_resampled
                    customers[prod_group][product] = resampled

        return df_group

    # def resample_bak(self, df_group: dict) -> dict:
    #     """
    #     Data Resampling
    #     :param df_group: time series dataset
    #     :return:
    #     """
    #     for group in df_group.values():
    #         for key, val in group.items():
    #             resampled = defaultdict(dict)
    #             for rule in self.time_rule:
    #                 val_dt = val.set_index(config.COL_DATETIME)
    #                 val_dt = val_dt.resample(rule=rule).sum()
    #                 if self.smooth_yn:
    #                     val_dt = self.smoothing(df=val_dt)
    #                 resampled[rule] = val_dt
    #             group[key] = resampled
    #
    #     return df_group

    def smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        for i, col in enumerate(df.columns):
            min_val = 0
            max_val = 0
            if self.smooth_method == 'quantile':
                min_val = df[col].quantile(self.smooth_rate)
                max_val = df[col].quantile(1 - self.smooth_rate)
            elif self.smooth_method == 'sigma':
                mean = np.mean(df[col].values)
                std = np.std(df[col].values)
                min_val = mean - 2 * std
                max_val = mean + 2 * std

            df[col] = np.where(df[col].values < min_val, min_val, df[col].values)
            df[col] = np.where(df[col].values > max_val, max_val, df[col].values)

        return df

    @staticmethod
    def split_sequence(df, n_steps_in, n_steps_out) -> tuple:
        """
        Split univariate sequence data
        :param df: Time series data
        :param n_steps_in:
        :param n_steps_out:
        :return:
        """
        data = df.astype('float32')
        x = []
        y = []
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(df):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, :]
            x.append(seq_x)
            y.append(seq_y)

        return np.array(x), np.array(y)