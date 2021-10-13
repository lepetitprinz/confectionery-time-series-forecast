import common.util as util

import os
import pickle

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Predict(object):
    estimators = {'rf': RandomForestRegressor,
                  'gb': GradientBoostingRegressor,
                  'et': ExtraTreesRegressor}

    def __init__(self, data_version: str, hrchy_lvl: int,
                 scaling_yn: bool, grid_search_yn: bool, save_obj_yn: bool):
        # Data Configuration
        self.data_version = data_version
        self.hrchy_lvl = hrchy_lvl

        # Prediction Option configuration
        self.scaling_yn = scaling_yn
        self.grid_search_yn = grid_search_yn
        self.save_obj_yn = save_obj_yn

    def predict(self, data, hrchy_code):
        # Load best estimator
        estimator = self.load_best_estimator(hrchy_code=hrchy_code)

        # Predict
        prediction = estimator.predict(data['x_test'])

        return prediction

    def load_best_estimator(self, hrchy_code: str):
        path = os.path.join(self.data_version + '_' + str(self.hrchy_lvl) + '_' + hrchy_code + '.pickle')
        f = open(path, 'rb')
        estimator = pickle.load(f)

        return estimator

