import numpy as np
import pandas as pd
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

    def __init__(self):
        # Feature instance
        self.n_feature = 5    # 52 weeks

        # Feature Importance Method instance
        self.method = 'dt'    # pca / lr(linear regression) / dt(decision tree) / rf(random forest)
        self.n_components = self.n_feature
        self.scaling_method = 'mnmx'    # std / mnmx

        # Weight applying method
        self.weight_apply_method = 'threshold'    # all / threshold / top_n
        self.weight_threshold = 0.2
        self.weight_top_n = 3

    def run(self, data) -> list:
        weights = self.generate_time_series_weight(data=data)

        return weights

    def generate_time_series_weight(self, data: pd.DataFrame):
        for cust_grp, cust_grp_df in data.groupby(by='cust_grp_cd'):
            for sku, sku_df in cust_grp_df.groupby(by='sku_cd'):
                sales = sku_df.sort_values(by=['yy', 'week', 'sales'])['sales']
                sales = sales.to_numpy().reshape(-1, 1)

                # Scale the dataset
                sales_scaled = self.scaling(data=sales)

                # Data window generator
                sales_window = self.window_generator(data=sales_scaled.ravel())

                # Generate weight
                weights = self.generate_featrue_importance(data=sales_window)

                # Weight apply method
                weights = self.apply_weight(weights=weights)

        return data

    def generate_featrue_importance(self, data) -> list:
        weights = []
        if self.method == 'pca':
            weight = self.pca(data=data)
        else:
            x = data[:, :-1]
            y = data[:, -1]
            weights = self.regression_estimate(estimator=self.method, x=x, y=y)
            weights = [round(weight, 2) for weight in weights]

        return weights

    def window_generator(self, data: np.array):
        sliced = []
        for i in range(len(data) - self.n_feature + 1):
            sliced.append(data[i: i+self.n_feature])

        return np.array(sliced)

    def scaling(self, data):
        # Instantiate scaler class
        scaler = None

        if self.scaling_method == 'std':
            scaler = StandardScaler()
        elif self.scaling_method == 'mnmx':
            scaler = MinMaxScaler()

        # Fit data and Transform it
        transform_data = scaler.fit_transform(data)

        return transform_data

    def regression_estimate(self, estimator, x, y):
        model = self.estimators[estimator]
        model.fit(x, y)

        if estimator == 'lr':
            importance = model.coef_
        else:
            importance = model.feature_importances_

        return importance

    def pca(self, data):
        # Instantiate pca method
        pca = PCA(n_components=self.n_components)

        # Determine transformed features
        pca.fit(data)

        # Get explained variance result
        importance = pca.explained_variance_ratio_

        return importance

    def apply_weight(self, weights: list):
        weights = [(i, weight) for i, weight in enumerate(weights)]

        if self.weight_apply_method == 'all':
            weights = weights
        elif self.weight_apply_method == 'threshold':
            weights = [(i, weight) for i, weight in weights if weight >= self.weight_threshold]
        elif self.weight_apply_method == 'top_n':
            weights = sorted(weights, key=lambda x: x[1], reverse=True)[:self.weight_top_n+1]

        return weights
