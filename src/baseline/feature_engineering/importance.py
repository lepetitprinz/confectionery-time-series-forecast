import common.config as config

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List, Any

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class FeatureImportance(object):
    estimators = {
        'lr': LinearRegression(),
        'dt': DecisionTreeRegressor(),
        'rf': RandomForestRegressor()
    }

    def __init__(self, item_mst, yy_week: pd.DataFrame, n_feature: int):
        self.yy_week = yy_week
        self.item_mst = item_mst

        # Feature instance
        self.n_feature = n_feature    # 52 weeks
        self.idx_to_week = {}

        # Feature importance method instance
        self.method = 'manual'    # manual / pca / lr(linear regression) / dt(decision tree) / rf(random forest)
        self.n_components = self.n_feature
        self.scaling_method = 'mnmx'    # std (standard deviation) / mnmx (min-max)

        # Weight applying method
        self.weight_top_n = 5    # Top N numbers
        self.feature_week_map = {}
        self.weight_threshold = 0.2
        self.weight_apply_method = 'top_n'    # all / threshold / top_n / manual

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.make_feature_idx_map()

        data = self.add_item_info(data=data)
        data = self.sum_by_upper_level(data=data)
        weights = self.generate_time_series_weight(data=data)

        return weights

    def make_feature_idx_map(self) -> None:
        feature_week = self.yy_week['week'].iloc[-self.n_feature:].copy()

        self.feature_week_map = {i: week for i, week in enumerate(feature_week)}

    def add_item_info(self, data: pd.DataFrame) -> pd.DataFrame:
        item_temp = deepcopy(self.item_mst)
        item_col = [col for col in item_temp.columns if 'nm' not in col]    # Item code list
        item_temp = item_temp[item_col]    # Filter item code data

        merged = pd.merge(data, item_temp, how='left', on=['sku_cd'])    # Merge item information

        return merged

    @staticmethod
    def sum_by_upper_level(data: pd.DataFrame):
        data_group = data.groupby(
            by=['division_cd', 'cust_grp_cd', 'biz_cd', 'line_cd', 'brand_cd', 'yy', 'week']
        ).sum().reset_index()

        return data_group

    def generate_time_series_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        cust_brand_weight = pd.DataFrame()
        for cust_grp, cust_grp_df in data.groupby(by='cust_grp_cd'):
            for brand, brand_df in cust_grp_df.groupby(by='brand_cd'):
                sales = self.fill_na_week(data=brand_df)

                # Reshape sales dataset
                sales = sales.to_numpy().reshape(-1, 1)

                # Scale the dataset
                sales_scaled = self.scaling(data=sales)

                # Data window generator
                sales_window = self.window_generator(data=sales_scaled.ravel())

                # Generate weight
                weights = self.generate_featrue_importance(data=sales_window)

                # Weight apply method
                weight_map = self.apply_weight(weights=weights)
                weight_df = self.make_weight_df(
                    weight=weight_map,
                    cust_grp=cust_grp,
                    brand=brand
                )
                cust_brand_weight = pd.concat([cust_brand_weight, weight_df], axis=0)

        return cust_brand_weight

    @staticmethod
    def make_weight_df(weight: dict, cust_grp, brand) -> pd.DataFrame:
        weight_df = pd.DataFrame(weight)
        weight_df['cust_grp_cd'] = cust_grp
        weight_df['brand_cd'] = brand

        return weight_df

    def fill_na_week(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[['yy', 'week', 'sales']].copy()
        merged = pd.merge(self.yy_week, data, how='left', on=['yy', 'week'])
        merged = merged.fillna(0)
        merged = merged.sort_values(by=['yy', 'week'])['sales']

        return merged

    def generate_featrue_importance(self, data: np.array) -> List[float]:
        if self.method == 'manual':
            weights = config.weights
        elif self.method == 'pca':
            weights = self.pca(data=data)
        else:
            x = data[:, :-1]
            y = data[:, -1]
            weights = self.regression_estimate(estimator=self.method, x=x, y=y)
            weights = [round(weight, 2) for weight in weights]

        return weights

    def window_generator(self, data: np.array) -> np.array:
        sliced = []
        for i in range(len(data) - self.n_feature):
            sliced.append(data[i: i + self.n_feature + 1])

        return np.array(sliced)

    def scaling(self, data: np.array) -> np.array:
        # Instantiate scaler class
        scaler = None

        if self.scaling_method == 'std':
            scaler = StandardScaler()
        elif self.scaling_method == 'mnmx':
            scaler = MinMaxScaler()

        # Fit data and Transform it
        transform_data = scaler.fit_transform(data)

        return transform_data

    def regression_estimate(self, estimator: str, x: np.array, y: np.array) -> List[int]:
        model = self.estimators[estimator]
        model.fit(x, y)

        if estimator == 'lr':
            importance = model.coef_
        else:
            importance = model.feature_importances_

        return importance

    def pca(self, data: np.array) -> List[int]:
        # Instantiate pca method
        pca = PCA(n_components=self.n_components)

        # Determine transformed features
        pca.fit(data)

        # Get explained variance result
        importance = pca.explained_variance_ratio_

        return importance

    def apply_weight(self, weights: list) -> Dict[str, List[Any]]:
        weights = [(i, weight) for i, weight in enumerate(weights)]

        if self.weight_apply_method == 'all':
            weights = weights
        elif self.weight_apply_method == 'threshold':
            weights = [(i, weight) for i, weight in weights if weight >= self.weight_threshold]
        elif self.weight_apply_method == 'top_n':
            weights = sorted(weights, key=lambda x: x[1], reverse=True)[:self.weight_top_n+1]
            weights = sorted(weights, key=lambda x: x[0])

        weights_map = {
            'week': [self.feature_week_map[idx] for idx, weight in weights],
            'weight': [weight for idx, weight in weights]
        }

        return weights_map
