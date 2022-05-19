import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


class FeatureImportance(object):
    def __init__(self):
        # Feature instance
        self.n_row = 105
        self.n_feature = 52    # 52 weeks

        # Feature Importance Method instance
        self.method = 'pca'    # pca / linear
        self.n_components = 'mle'
        self.scaling_method = 'mnmx'    # std / mnmx

        # Weight applying method
        self.weight_apply_method = 'all'    # all / threshold / top_n

    def run(self, data) -> list:
        data = self.make_features(data=data)

        # Calculate feature importance
        importance = []
        if self.method == 'pca':
            data = self.scaling(data=data)
            importance = self.pca(data=data)

        elif self.method == 'linear':
            importance = self.linear_regression(data=data)

        return importance

    def make_features(self, data: pd.DataFrame):


        return data

    def window_generator(self, data: pd.DataFrame):
        data = data.to_numpy()

        sliced = []
        for i in range(len(data) - self.n_feature + 1):
            sliced.append(data[i: i+self.n_feature])

        return sliced

    def scaling(self, data):
        # Instantiate scaler class
        scaler = None
        if self.scaling_method == 'std':
            scaler = StandardScaler()
        elif self.scaling_method == 'mnmx':
            scaler = MinMaxScaler()

        #
        transform_data = scaler.fit_transform(data)

        return transform_data

    def pca(self, data):
        # Instantiate pca method
        pca = PCA(n_components=self.n_components)

        # Determine transformed features
        pca.fit(data)

        importance = pca.explained_variance_ratio_

        return importance

    def linear_regression(self, data: dict):
        model = LinearRegression()
        model.fit(data['X'], data['y'])

        # Feature importance
        importance = model.coef_

        return importance

    def apply_weight(self, data, weights):
        if self.weight_apply_method == 'all':
            pass
        elif self.weight_apply_method == 'threshold':
            pass
        elif self.weight_apply_method == 'top_n':
            pass