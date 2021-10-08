import common.util as util
import common.config as config
from simulation.model.Algorithm import Algorithm

import os
import pickle

from sklearn.model_selection import GridSearchCV


class Train(object):
    def __init__(self, mst_info: dict, hrchy_lvl: int, grid_search_yn: bool, param_grid={}):
        # Class Configuration
        self.algorithm = Algorithm()

        self.hrchy_lvl = hrchy_lvl

        # Training configuration
        self.grid_search_yn = grid_search_yn

        # Algorithm Configuration
        self.param_best = mst_info['param_grid']
        self.param_grid = param_grid
        self.model_fn = {'rf': self.algorithm.random_forest,
                         'extr': self.algorithm.e}

    def train(self, df):
        if self.grid_search_yn:
            pass
        else:
            scores = util.hrchy_recursion(hrchy_lvl=self.hrchy_lvl-1,
                                      fn=self.resample,
                                      df=df)

    def grid_search_cross_validation(self, data: dict, regr: str):
        # Grid search cross validation
        regr_best = self.get_best_hyper_param(x=train['x'], y=train['y'],
                                               regr=regr,
                                               regressors=self.REGRESSORS,
                                               param_grids=self.param_grids)
        model_bests[model_key] = regr_best
        regr_bests[type_key] = model_bests

        return regr_bests

    @staticmethod
    def get_best_hyper_param(x, y, regressors, param_grids, regr, scoring='neg_root_mean_squared_error'):
        # Select regressor algorithm
        regressor = regressors[regr]

        # Define parameter grid
        param_grid = param_grids[regr]

        # Initialize Grid Search object
        gscv = GridSearchCV(estimator=regressor, param_grid=param_grid,
                            scoring=scoring, n_jobs=1, cv=5, verbose=1)

        # Fit gscv
        print(f'Tuning {regr}')
        gscv.fit(x, y)

        # Get best paramters and score
        best_params = gscv.best_params_
        # best_score = gscv.best_score_

        # Update regressor paramters
        regressor.set_params(**best_params)

        return regressor

    def save_best_params(self, regr: str, regr_bests: dict, type_apply: str):
        for type_key, type_val in regr_bests.items():
            for model_key, model_val in type_val.items():
                best_params = model_val.get_params()
                f = open(os.path.join(self.path_res_pred, type_apply, type_key,
                                      regr + '_params_' + model_key + '.pickle'), 'wb')
                pickle.dump(best_params, f)
                f.close()