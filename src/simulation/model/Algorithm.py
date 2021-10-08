import ast
import numpy as np
from sklearn.metrics import mean_squared_error

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Algorithm(object):
    # Random Forest
    @staticmethod
    def random_forest(data: dict, cfg: dict):
        """

        """
        regr = RandomForestRegressor(
            n_estimators=ast.literal_eval(cfg['n_estimators']),
            criterion=cfg['criterion'],
            max_features=cfg['max_features']
        )
        regr.fit(data['x_train'], data['y_train'])

        yhat = regr.predict(data['x_test'])
        err = mean_squared_error(data['y_test'], yhat, squared=True)

        return err

    # Random Forest
    @staticmethod
    def gradient_boost(data: dict, cfg: dict):
        """

        """
        regr = GradientBoostingRegressor(
            n_estimators=ast.literal_eval(cfg['n_estimators']),
            criterion=cfg['criterion'],
            max_features=cfg['max_features']
        )
        regr.fit(data['x_train'], data['y_train'])

        yhat = regr.predict(data['x_test'])
        err = mean_squared_error(data['y_test'], yhat, squared=True)

        return err
