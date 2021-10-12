from simulation.model.Algorithm import Algorithm

import os
import pickle


class Predict(object):
    estimators = {'rf': Algorithm.random_forest,
                  'gb': Algorithm.gradient_boost,
                  'et': Algorithm.extra_trees}

    def __init__(self, data_version: str, mst_info: dict, hrchy_lvl: int):
        # Data Configuration
        self.data_version = data_version
        self.hrchy_lvl = hrchy_lvl

    def prediction(self, data, hrchy_code):
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

