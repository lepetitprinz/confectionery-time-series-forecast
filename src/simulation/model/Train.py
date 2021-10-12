import common.util as util
import common.config as config
from simulation.model.Algorithm import Algorithm

import os
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class Train(object):
    estimators = {'rf': Algorithm.random_forest,
                  'gb': Algorithm.gradient_boost,
                  'et': Algorithm.extra_trees}

    def __init__(self, data_version: str, hrchy_lvl: int, algorithms: list, parameters: dict,
                 grid_search_yn: bool, save_obj_yn: bool):
        # Data Configuration
        self.data_version = data_version
        self.hrchy_lvl = hrchy_lvl

        # Train Option configuration
        self.scoring = 'neg_root_mean_squared_error'
        self.cv = 5
        self.grid_search_yn = grid_search_yn
        self.save_obj_yn = save_obj_yn

        # Algorithm Configuration
        self.algorithms = algorithms
        self.parameters = parameters
        self.param_grids = config.PARAM_GRIDS_SIM

    def train(self):
        pass

    def train_best(self, data, hrchy_code, verbose=False):
        best_model = self.evaluation(
            data=data,
            estimators=self.algorithms,
            grid_search_yn=self.grid_search_yn,
            scoring=self.scoring,
            cv=self.cv,
            verbose=verbose
        )
        if self.save_obj_yn:
            self.save_best_result(estimator=best_model, hrchy_code=hrchy_code)

        return best_model

    def evaluation(self, data, estimators: list, grid_search_yn: bool, verbose: bool,
                   scoring='neg_root_mean_squared_error', cv=5):
        # Execute grid search cross validation
        results = []
        for estimator in estimators:
            if grid_search_yn:
                score, params = self.grid_search_cv(
                    data=data,
                    estimator=self.estimators[estimator],
                    param_grid=self.parameters['param_grids'][estimator],
                    scoring=scoring,
                    cv=cv,
                    verbose=verbose
                )
            else:
                score, params = self.cross_validation(
                    data=data,
                    estimator=self.estimators[estimator],
                    param_grid=self.parameters['param_best'][estimator],
                    scoring=scoring,
                    cv=cv,
                    verbose=verbose
                )
            # append each result
            results.append((estimator, params, score))

        # Get best model
        results = sorted(results, key=lambda x: x[-1])  # Sort by score
        best_model = results[0]

        return best_model

    @staticmethod
    def grid_search_cv(data, estimator, param_grid: dict, scoring, cv: int, verbose: bool):
        gsc = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        result = gsc.fit(data['x_train'], data['y_train'])

        if verbose:
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            for test_mean, train_mean, param in zip(
                    result.cv_results_['mean_test_score'],
                    result.cv_results_['mean_train_score'],
                    result.cv_results_['params']):
                print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

        return result.best_score_, result.best_params_

    @staticmethod
    def cross_validation(data: dict, estimator, param_grid: dict, scoring: str, cv: int, verbose: bool):
        regr = estimator()
        regr.set_params(**param_grid)
        scores = cross_val_score(regr, data['x_train'], data['y_train'],
                                 scoring=scoring, cv=cv)

        score = sum(scores) / len(scores)

        if verbose:
            print(f'Estimator: {type(regr).__name__}, Score: {score}')

        return score, param_grid

    def save_best_result(self, estimator, hrchy_code: str):
        f = open(os.path.join(
            self.data_version + '_' + str(self.hrchy_lvl) + '_' + hrchy_code + '.pickle'), 'wb')
        pickle.dump(estimator, f)
        f.close()
