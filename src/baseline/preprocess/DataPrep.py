from baseline.analysis.Decomposition import Decomposition
import common.util as util

from copy import deepcopy
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import timedelta

from sklearn.impute import KNNImputer


class DataPrep(object):
    DROP_COLS_DATA_PREP = ['division_cd', 'seq', 'from_dc_cd', 'unit_price', 'create_date']
    STR_TYPE_COLS = ['cust_cd', 'sku_cd']

    def __init__(self, date: dict, cust: pd.DataFrame, division: str, common: dict,
                 hrchy: dict, exec_cfg: dict):
        # Dataset configuration
        self.exec_cfg = exec_cfg
        self.division = division
        self.cust = cust
        self.common = common
        self.date_col = common['date_col']
        self.resample_rule = common['resample_rule']
        self.col_agg_map = {
            'sum': common['agg_sum'].split(','),
            'avg': common['agg_avg'].split(',')
        }
        self.date_range = pd.date_range(
            start=date['date_from'],
            periods=52 * ((int(date['date_to'][:4]) - int(date['date_from'][:4])) + 1) + 1,
            freq=common['resample_rule']
        )

        # Hierarchy configuration
        self.hrchy = hrchy
        self.hrchy_level = hrchy['lvl']['cust'] + hrchy['lvl']['item'] - 1

        # Execute option
        self.imputer = 'knn'
        self.outlier_method = 'std'
        self.quantile_range = 0.02

    def preprocess(self, data: pd.DataFrame, exg: dict) -> dict:
        # ------------------------------- #
        # 1. Preprocess sales dataset
        # ------------------------------- #
        # convert data type
        for col in self.STR_TYPE_COLS:
            data[col] = data[col].astype(int).astype(str)

        # Mapping: cust_cd -> cust_grp_cd
        data = pd.merge(data, self.cust, on=['cust_cd'], how='left')
        data['cust_grp_cd'] = data['cust_grp_cd'].fillna('-')

        # ------------------------------- #
        # 2. Preprocess Exogenous dataset
        # ------------------------------- #
        exg_all = util.prep_exg_all(data=exg['all'])

        # preprocess exogenous(partial) data
        # exg_partial = util.prep_exg_partial(data=exg['partial'])

        # ------------------------------- #
        # 3. Preprocess merged dataset
        # ------------------------------- #
        # Merge sales data & exogenous(all) data
        data = pd.merge(data, exg_all, on=self.date_col, how='left')

        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Grouping
        # data_group = self.group(data=data)
        data_group = util.group(hrchy=self.hrchy['apply'], hrchy_lvl=self.hrchy_level, data=data)

        # Decomposition
        if self.exec_cfg['decompose_yn']:
            decompose = Decomposition(
                common=self.common,
                division=self.division,
                hrchy=self.hrchy,
                date_range=self.date_range
            )

            util.hrchy_recursion(
                hrchy_lvl=self.hrchy_level,
                fn=decompose.decompose,
                df=data_group
            )

            decompose.dao.session.close()

        # Resampling
        data_resample = util.hrchy_recursion(
            hrchy_lvl=self.hrchy_level,
            fn=self.resample,
            df=data_group
        )

        return data_resample

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop unnecessary columns
        df = df.drop(columns=self.__class__.DROP_COLS_DATA_PREP, errors='ignore')

        # Convert unit code
        if self.division == 'SELL_IN':
            conditions = [df['unit_cd'] == 'EA ',
                          df['unit_cd'] == 'BOL',
                          df['unit_cd'] == 'BOX']

            values = [df['box_ea'], df['box_bol'], 1]
            unit_map = np.select(conditions, values)
            df['qty'] = df['qty'].to_numpy() / unit_map

            df = df.drop(columns=['box_ea', 'box_bol'], errors='ignore')

        # convert to datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col], format='%Y%m%d')
        df = df.set_index(keys=[self.date_col])

        # add noise feature
        # df = self.add_noise_feat(df=df)

        return df

    def group(self, data, cd=None, lvl=0) -> dict:
        grp = {}
        col = self.hrchy[lvl]

        code_list = None
        if isinstance(data, pd.DataFrame):
            code_list = list(data[col].unique())

        elif isinstance(data, dict):
            code_list = list(data[cd][col].unique())

        if lvl < self.hrchy_level:
            for code in code_list:
                sliced = None
                if isinstance(data, pd.DataFrame):
                    sliced = data[data[col] == code]
                elif isinstance(data, dict):
                    sliced = data[cd][data[cd][col] == code]
                result = self.group(data={code: sliced}, cd=code, lvl=lvl + 1)
                grp[code] = result

        elif lvl == self.hrchy_level:
            temp = {}
            for code in code_list:
                sliced = None
                if isinstance(data, pd.DataFrame):
                    sliced = data[data[col] == code]
                elif isinstance(data, dict):
                    sliced = data[cd][data[cd][col] == code]
                temp[code] = sliced

            return temp

        return grp

    def resample(self, df: pd.DataFrame):
        df_sum_resampled = self.resample_by_agg(df=df, agg='sum')
        df_avg_resampled = self.resample_by_agg(df=df, agg='avg')

        # Concatenate aggregation
        df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        if len(df_resampled.index) != len(self.date_range):
            df_resampled = self.fill_missing_date(df=df_resampled)

        # Add data level
        df_resampled = self.add_data_level(org=df, resampled=df_resampled)

        return df_resampled

    def resample_by_agg(self, df, agg: str):
        resampled = pd.DataFrame()
        col_agg = set(df.columns).intersection(set(self.col_agg_map[agg]))
        if len(col_agg) > 0:
            resampled = df[self.col_agg_map[agg]]
            resampled = resampled.resample(rule=self.resample_rule).sum()  # resampling
            resampled = resampled.fillna(value=0)  # fill NaN

        return resampled

    def fill_missing_date(self, df):
        idx_add = list(set(self.date_range) - set(df.index))
        data_add = np.zeros((len(idx_add), df.shape[1]))
        df_add = pd.DataFrame(data_add, index=idx_add, columns=df.columns)
        df = df.append(df_add)
        df = df.sort_index()

        return df

    def add_data_level(self, org, resampled):
        cols = self.hrchy[:self.hrchy_level + 1]
        data_level = org[cols].iloc[0].to_dict()
        data_lvl = pd.DataFrame(data_level, index=resampled.index)
        df_resampled = pd.concat([resampled, data_lvl], axis=1)

        return df_resampled

    def impute_data(self, df: pd.DataFrame, feat: str):
        feature = deepcopy(df[feat])
        if self.imputer == 'knn':
            feature = np.where(feature.values == 0, np.nan, feature.values)
            feature = feature.reshape(1, -1)
            imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(feature)
            feature = imputer.transform(feature)
            feature = feature.ravel()


        elif self.imputer == 'before':
            for i in range(1, len(feature)):
                if feature[i] == 0:
                    feature[i] = feature[i-1]

        elif self.imputer == 'avg':
            for i in range(1, len(feature)-1):
                if feature[i] == 0:
                    feature[i] = (feature[i-1] + feature[i+1]) / 2

        df[feat] = feature

        return df

    def remove_outlier(self, df: pd.DataFrame, feat: str):
        feature = deepcopy(df[feat])
        lower, upper = 0, 0

        if self.outlier_method == 'std':
            feature = feature.values
            mean = np.mean(feature)
            std = np.std(feature)
            cut_off = std * 3    # 99.7%
            lower = mean - cut_off
            upper = mean + cut_off

        elif self.outlier_method == 'quantile':
            lower = feature.quantile(self.quantile_range)
            upper = feature.quantile(1 - self.quantile_range)

        feature = np.where(feature < lower, lower, feature)
        feature = np.where(feature > upper, upper, feature)

        df[feat] = feature

        return df

    @staticmethod
    def make_seq_to_cust_map(df: pd.DataFrame):
        seq_to_cust = df[['seq', 'cust_cd']].set_index('seq').to_dict('index')

        return seq_to_cust

    # def add_noise_feat(self, df: pd.DataFrame) -> pd.DataFrame:
    #     vals = df[self.target_col].values * 0.05
    #     vals = vals.astype(int)
    #     vals = np.where(vals == 0, 1, vals)
    #     vals = np.where(vals < 0, vals * -1, vals)
    #     noise = np.random.randint(-vals, vals)
    #     df['exo'] = df[self.target_col].values + noise
    #
    #     return df