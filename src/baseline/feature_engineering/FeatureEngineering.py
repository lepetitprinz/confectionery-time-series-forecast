from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


class FeatureEngineering(object):
    def __init__(self, common: dict):
        self.target_col = self.target_col['dict']
        self.method = 'corr'    # corr / feat_it
        self.n_feature_to_select = 3

    def feature_selection(self, data):
        pass
    # Correlation

    def feature_importance(self, x, y):
        rfe = RFE(
            RandomForestRegressor(
                n_estimators=500,
                random_state=1,
            ),
            n_features_to_select=self.n_feature_to_select
        )
        fit = rfe.fit(x, y)
        print(fit.support_)
