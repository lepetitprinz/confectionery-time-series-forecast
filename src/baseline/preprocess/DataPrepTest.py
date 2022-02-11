import datetime

from baseline.analysis.Decomposition import Decomposition
from baseline.feature_engineering.FeatureEngineering import FeatureEngineering
import common.util as util
import common.config as config

import numpy as np
import pandas as pd
from math import ceil
from copy import deepcopy
from typing import Tuple, Union
from sklearn.impute import KNNImputer


class DataPrepTest(object):
    DROP_COLS_DATA_PREP = ['division_cd', 'seq', 'from_dc_cd', 'unit_price', 'create_date']
    STR_TYPE_COLS = ['cust_grp_cd', 'sku_cd']

    def __init__(self, date: dict, common: dict, hrchy: dict, data_cfg: dict, exec_cfg: dict):
        # Dataset configuration
        self.common = common
        self.data_cfg = data_cfg
        self.exec_cfg = exec_cfg
        self.resample_rule = data_cfg['cycle']
        self.col_agg_map = {
            'sum': common['agg_sum'].split(','),
            'avg': common['agg_avg'].split(',')
        }
        self.hist_date_range = pd.date_range(
            start=date['history']['from'],
            end=date['history']['to'],
            freq=common['resample_rule']
        )
        self.date = date
        self.date_length = len(self.hist_date_range)

        self.exg_map = config.EXG_MAP    # Exogenous variable map
        self.key_col = ['cust_grp_cd', 'sku_cd']

        # Hierarchy configuration
        self.hrchy = hrchy
        self.hrchy_level = hrchy['lvl']['cust'] + hrchy['lvl']['item'] - 1
        self.hrchy_cnt = 0

        # Data threshold
        self.decimal_point = 2
        self.exec_date = None
        self.rm_lvl_cnt = 0
        self.tot_sp1_sku_cnt = 0
        self.rm_sp1_sku_cnt = 0
        self.threshold_cnt = int(self.common['filter_threshold_cnt'])
        self.threshold_recent = int(self.common['filter_threshold_recent'])
        self.threshold_sku_period = datetime.timedelta(
            days=int(self.common['filter_threshold_sku_recent']) * 7
        )

        # Execute option
        self.imputer = 'knn'
        self.outlier_method = 'std'
        self.quantile_range = 0.02
        self.noise_rate = 0.1
        self.sigma = float(common['outlier_sigma'])

    def preprocess(self, data: pd.DataFrame, exg: pd.DataFrame) -> Tuple[dict, list, int]:
        # Preprocess sales dataset
        if not self.exec_cfg['decompose_yn']:
            exg_list = list(idx.lower() for idx in exg['idx_cd'].unique())
        else:
            exg_list = []

        # Initiate feature engineering class
        fe = FeatureEngineering(common=self.common, exg_list=exg_list)

        # convert data type
        for col in self.STR_TYPE_COLS:
            data[col] = data[col].astype(str)

        # convert datetime column
        data[self.common['date_col']] = data[self.common['date_col']].astype(np.int64)

        # Preprocess Exogenous dataset
        if not self.exec_cfg['decompose_yn']:
            exg = util.prep_exg_all(data=exg)

            # Merge sales data & exogenous(all) data
            data = self.merge_exg(data=data, exg=exg)

        # preprocess sales dataset
        data = self.conv_data_type(df=data)

        # Feature engineering
        if self.exec_cfg['feature_selection_yn']:
            data, exg_list = fe.feature_selection(data=data)

        # Group by data level hierarchy
        data_group, self.hrchy_cnt = util.group(
            hrchy=self.hrchy['apply'],
            hrchy_lvl=self.hrchy_level,
            data=data
        )

        # Filter threshold SKU based on recent sales
        if self.exec_cfg['filter_threshold_recent_sku_yn']:
            # Execute date
            exec_date = self.date['history']['to']
            self.exec_date = datetime.datetime.strptime(exec_date, '%Y%m%d') + datetime.timedelta(days=1)

            data_group = util.hrchy_recursion_with_none(
                hrchy_lvl=self.hrchy_level,
                fn=self.filter_threshold_recent_sku,
                df=data_group
            )
            print("-----------------------------")
            print(f"Total SKU: {self.tot_sp1_sku_cnt}")
            print(f"Removed SKU: {self.rm_sp1_sku_cnt}")
            print(f"Applied SKU: {self.tot_sp1_sku_cnt - self.rm_sp1_sku_cnt}")
            print("-----------------------------")

        # Representative sampling
        if self.exec_cfg['representative_sampling_yn']:
            # Columns of representative sampling are added to aggregates option instance
            exg_add_list = [self.common['target_col'] + '_' + col for col in fe.repr_sampling_agg_list]
            exg_list += exg_add_list
            self.col_agg_map['avg'].extend(exg_add_list)

            # add columns of representative sampling
            data_group = util.hrchy_recursion_with_none(
                hrchy_lvl=self.hrchy_level,
                fn=fe.repr_sampling,
                df=data_group
            )

        # Decomposition
        if self.exec_cfg['decompose_yn']:
            decompose = Decomposition(
                common=self.common,
                hrchy=self.hrchy,
                date=self.date
            )

            decomposed = util.hrchy_recursion_with_none(
                hrchy_lvl=self.hrchy_level,
                fn=decompose.decompose,
                df=data_group
            )

            return decomposed, exg_list, self.hrchy_cnt

        # Resampling
        data_resample = util.hrchy_recursion_with_none(
            hrchy_lvl=self.hrchy_level,
            fn=self.resample,
            df=data_group
        )

        print('-----------------------------------')
        print(f"Total data level counts: {self.hrchy_cnt}")
        print(f"Removed data level counts: {self.rm_lvl_cnt}")
        print(f"Applying data level counts: {self.hrchy_cnt - self.rm_lvl_cnt}")
        print('-----------------------------------')
        self.hrchy_cnt -= self.rm_lvl_cnt

        # Columns of rolling statistics are added to aggregates option instance
        if self.exec_cfg['rolling_statistics_yn']:
            exg_add_list = [self.common['target_col'] + '_' + col for col in fe.rolling_agg_list]
            exg_list += exg_add_list

        return data_resample, exg_list, self.hrchy_cnt

    # Merge exogenous variables
    def merge_exg(self, data: pd.DataFrame, exg: dict) -> pd.DataFrame:
        cust_grp_list = list(data['cust_grp_cd'].unique())

        merged = pd.DataFrame()
        for cust_grp in cust_grp_list:
            temp = data[data['cust_grp_cd'] == cust_grp]
            temp = pd.merge(temp, exg[self.exg_map.get(cust_grp, '999')], on=self.common['date_col'], how='left')
            merged = pd.concat([merged, temp])

        return merged

    def conv_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop unnecessary columns
        df = df.drop(columns=self.__class__.DROP_COLS_DATA_PREP, errors='ignore')
        # df['unit_cd'] = df['unit_cd'].str.replace(' ', '')
        # Convert unit code
        # if self.data_cfg['division'] == 'SELL_OUT':
        #     conditions = [df['unit_cd'] == 'EA',
        #                   df['unit_cd'] == 'BOL',
        #                   df['unit_cd'] == 'BOX']
        #
        #     values = [df['box_ea'], df['box_bol'], 1]
        #     unit_map = np.select(conditions, values)
        #     df['qty'] = df['qty'].to_numpy() / unit_map
        #
        #     df = df.drop(columns=['box_ea', 'box_bol'], errors='ignore')

        # convert to datetime
        df[self.common['date_col']] = pd.to_datetime(df[self.common['date_col']], format='%Y%m%d')
        df = df.set_index(keys=[self.common['date_col']])

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

    def resample(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        df_sum_resampled = self.resample_by_agg(df=df, agg='sum')
        df_avg_resampled = self.resample_by_agg(df=df, agg='avg')

        # Concatenate aggregation
        # STR_TYPE_COLS = ['cust_grp_cd', 'sku_cd']
        df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        if self.exec_cfg['filter_threshold_cnt_yn']:
            if len(df_resampled[df_resampled['qty'] != 0]) < self.threshold_cnt:
                self.rm_lvl_cnt += 1
                return None

        # Fill empty sales to zeros
        if len(df_resampled.index) != len(self.hist_date_range):
            # missed_rate = self.check_missing_data(df=df_resampled)
            df_resampled = self.fill_missing_date(df=df_resampled)

        # Filter data level under threshold recent week
        if self.exec_cfg['filter_threshold_recent_yn']:
            if self.filter_threshold_recent(data=df_resampled, col=self.common['target_col']):
                self.rm_lvl_cnt += 1
                return None

        # Remove forward empty sales
        if self.exec_cfg['rm_fwd_zero_sales_yn']:
            df_resampled = self.rm_fwd_zero_sales(df=df_resampled, feat=self.common['target_col'])

        # Todo : New Function
        # Rolling statistics of time series
        if self.exec_cfg['rolling_statistics_yn']:
            fe = FeatureEngineering(common=self.common)
            df_resampled = fe.rolling(df=df_resampled, feat=self.common['target_col'])

        # Remove outlier
        if self.exec_cfg['rm_outlier_yn']:
            df_resampled = self.remove_outlier(df=df_resampled, feat=self.common['target_col'])

        # Data imputation
        if self.exec_cfg['data_imputation_yn']:
            df_resampled = self.impute_data(df=df_resampled, feat=self.common['target_col'])

        # Add data level
        df_resampled = self.add_data_level(org=df, resampled=df_resampled)

        return df_resampled

    def filter_threshold_recent(self, data: pd.DataFrame, col: str) -> bool:
        check = False
        data_length = len(data[col])
        data_trimmed_length = len(np.trim_zeros(data[col].to_numpy(), trim='b'))
        diff = data_length - data_trimmed_length
        if diff > self.threshold_recent:
            check = True

        return check

    def filter_threshold_recent_sku(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        df['date'] = df.index
        last_day_by_sku = df[['sku_cd', 'date']].groupby(by=['sku_cd']).max().reset_index()
        last_day_by_sku['diff'] = self.exec_date - last_day_by_sku['date']
        all_sku = list(df['sku_cd'].unique())
        valid_sku = list(last_day_by_sku[last_day_by_sku['diff'] < self.threshold_sku_period]['sku_cd'])

        self.tot_sp1_sku_cnt += len(all_sku)
        self.rm_sp1_sku_cnt += (len(all_sku) - len(valid_sku))

        # Filtering sku
        result = df[df['sku_cd'].isin(valid_sku)]

        if len(result) == 0:
            self.hrchy_cnt -= 1
            result = None

        return result

    @staticmethod
    def rm_fwd_zero_sales(df: pd.DataFrame, feat: str) -> pd.DataFrame:
        feature = df[feat].to_numpy()
        feature = np.trim_zeros(feature, trim='f')
        df_trim = df.iloc[len(df) - len(feature):, :]

        return df_trim

    def check_missing_value(self, df: pd.DataFrame) -> float:
        # df_sum_resampled = self.resample_by_agg(df=df, agg='sum')
        # df_avg_resampled = self.resample_by_agg(df=df, agg='avg')
        #
        # # Concatenate aggregation
        # df_resampled = pd.concat([df_sum_resampled, df_avg_resampled], axis=1)

        # Check and add dates when sales does not exist
        missed_rate = 0
        if len(df.index) != len(self.hist_date_range):
            missed_rate = self.check_missing_data(df=df)

        return missed_rate

    def check_missing_data(self, df: pd.DataFrame) -> Tuple[int, float]:
        tot_len = len(self.hist_date_range)
        missed = tot_len - len(df.index)
        exist = tot_len - missed
        missed_rate = 100 - round((missed / tot_len) * 100, 1)

        return exist, missed_rate

    def resample_by_agg(self, df: pd.DataFrame, agg: str) -> pd.DataFrame:
        resampled = pd.DataFrame()
        col_agg = set(df.columns).intersection(set(self.col_agg_map[agg]))
        if len(col_agg) > 0:
            resampled = df[col_agg]
            if agg == 'sum':
                resampled = resampled.resample(rule=self.resample_rule).sum().round(self.decimal_point)  # resampling
            elif agg == 'avg':
                resampled = resampled.resample(rule=self.resample_rule).mean().round(self.decimal_point)
            resampled = resampled.fillna(value=0)  # fill NaN

        return resampled

    def fill_missing_date(self, df: pd.DataFrame) -> pd.DataFrame:
        idx_add = list(set(self.hist_date_range) - set(df.index))
        data_add = np.zeros((len(idx_add), df.shape[1]))
        df_add = pd.DataFrame(data_add, index=idx_add, columns=df.columns)
        df = df.append(df_add)
        df = df.sort_index()

        return df

    def add_data_level(self, org: pd.DataFrame, resampled: pd.DataFrame) -> pd.DataFrame:
        cols = self.hrchy['apply'][:self.hrchy_level + 1]
        data_level = org[cols].iloc[0].to_dict()
        data_lvl = pd.DataFrame(data_level, index=resampled.index)
        df_resampled = pd.concat([resampled, data_lvl], axis=1)

        return df_resampled

    def impute_data(self, df: pd.DataFrame, feat: str) -> pd.DataFrame:
        feature = deepcopy(df[feat])
        if self.imputer == 'knn':
            if len(feature) > 0:
                feature = np.where(feature.values == 0, np.nan, feature.values)
                feature = feature.reshape(-1, 1)
                imputer = KNNImputer(n_neighbors=3)
                feature = imputer.fit_transform(feature)
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

    def remove_outlier(self, df: pd.DataFrame, feat: str) -> pd.DataFrame:
        feature = deepcopy(df[feat])
        lower, upper = 0, 0

        if self.outlier_method == 'std':
            feature = feature.values
            mean = np.mean(feature)
            std = np.std(feature)
            cut_off = std * self.sigma   # 99.7%
            lower = mean - cut_off
            upper = mean + cut_off

        elif self.outlier_method == 'quantile':
            lower = feature.quantile(self.quantile_range)
            upper = feature.quantile(1 - self.quantile_range)

        # feature = np.where(feature < 0, 0, feature)
        feature = np.where(feature < lower, lower, feature)    # Todo:
        feature = np.where(feature > upper, upper, feature)

        df[feat] = feature

        return df

    @staticmethod
    def make_seq_to_cust_map(df: pd.DataFrame) -> dict:
        seq_to_cust = df[['seq', 'cust_cd']].set_index('seq').to_dict('index')

        return seq_to_cust

    def add_noise_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        feature = deepcopy(df[self.common['target_col']])

        # Calculate mean, max, min
        feat_mean = feature.mean()
        feat_max = feature.max()
        # feat_min = feature.min()

        # Set noise range
        rand_max = ceil(feat_max * self.noise_rate)
        # rand_min = round(feat_min * self.noise_rate)
        rand_norm = np.random.rand(self.date_length)

        rand_list = np.random.randint(0, rand_max, self.date_length) + rand_norm
        feat_add_noise = feature + rand_list
        feat_del_noise = feature - rand_list

        if feat_max > 2:
            values = np.where(feature >= feat_mean, feat_add_noise, feat_del_noise)
        else:
            values = feature + rand_norm

        df[self.common['target_col']] = values

        return df

    @staticmethod
    def ravel_df(hrchy: list, df: pd.DataFrame) -> np.array:
        result = df.to_numpy()

        return result

    def conv_decomposed_list(self, data: list) -> pd.DataFrame:
        cols = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd',
                'yymmdd', 'org_val', 'trend_val', 'seasonal_val', 'resid_val']

        df = pd.DataFrame(data, columns=cols)
        df['project_cd'] = self.common['project_cd']
        df['data_vrsn_cd'] = self.date['history']['from'] + '-' + self.date['history']['to']
        df['division_cd'] = self.data_cfg['division']
        df['hrchy_lvl_cd'] = self.hrchy['key'][:-1]
        df['seq'] = [str(i+1).zfill(10) for i in range(len(df))]
        df['seq'] = df['yymmdd'] + '-' + df['seq']
        df['create_user_cd'] = 'SYSTEM'
        df = df[['project_cd', 'data_vrsn_cd', 'division_cd', 'hrchy_lvl_cd', 'seq'] + cols + ['create_user_cd']]

        return df
