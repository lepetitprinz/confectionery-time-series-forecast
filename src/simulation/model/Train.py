import common.util as util
import common.config as config

import os
import pickle

# Algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class Train(object):
    estimators = {'rf': RandomForestRegressor,
                  'gb': GradientBoostingRegressor,
                  'et': ExtraTreesRegressor}

    def __init__(self, data_version: str, division: str, hrchy_lvl: int, common, algorithms: list, parameters: dict,
                 scaling_yn: bool, grid_search_yn: bool, save_obj_yn: bool):
        # Data Configuration
        self.data_version = data_version
        self.division = division
        self.hrchy_lvl = hrchy_lvl
        self.target_col = common['target_col']

        # Train Option configuration
        self.scoring = 'neg_root_mean_squared_error'
        self.cv = 5
        self.scaling_yn = scaling_yn
        self.grid_search_yn = grid_search_yn
        self.save_obj_yn = save_obj_yn
        self.verbose = False

        # Algorithm Configuration
        self.algorithms = algorithms
        self.parameters = parameters
        self.param_grids = config.PARAM_GRIDS_SIM

    def train(self, data):
        util.hrchy_recursion_with_key(
            hrchy_lvl=self.hrchy_lvl - 1,
            fn=self.train_best,
            df=data
        )
        print("Training is finished.")

    def train_best(self, hrchy_code, data):
        # Split dataset
        data_split = self.split_data(data=data)

        # Scaling
        if self.scaling_yn:
            scaler, x_scaled = self.scaling(data=data_split['x_train'])
            data_split['x_train'] = x_scaled
            self.save_scaler(scaler=scaler, hrchy_code=hrchy_code)

        best_model = self.evaluation(
            data=data_split,
            estimators=self.algorithms,
            grid_search_yn=self.grid_search_yn,
            scoring=self.scoring,
            cv=self.cv,
            verbose=self.verbose
        )
        if self.save_obj_yn:
            self.save_best_model(estimator=best_model, hrchy_code=hrchy_code)

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
            results.append([estimator, params, score])

        # Get best model
        results = sorted(results, key=lambda x: x[-1], reverse=True)  # Sort by score
        best_model = results[0]

        est_fit = self.fit_model(
            data=data,
            model=self.estimators[best_model[0]],
            params=best_model[1]
        )
        best_model.append(est_fit)

        return best_model

    @staticmethod
    def fit_model(data, model, params):
        estimator = model().set_params(**params)
        estimator.fit(data['x_train'], data['y_train'])

        return estimator

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
            print(f"Best: {result.best_score_} using {result.best_params_}")
            for test_mean, train_mean, param in zip(
                    result.cv_results_['mean_test_score'],
                    result.cv_results_['mean_train_score'],
                    result.cv_results_['params']):
                print(f"Train: {train_mean} // Test: {test_mean} with: {param}")

        return result.best_score_, result.best_params_

    @staticmethod
    def cross_validation(data: dict, estimator, param_grid: dict, scoring: str, cv: int, verbose: bool):
        regr = estimator()
        regr.set_params(**param_grid)
        scores = cross_val_score(regr, data['x_train'], data['y_train'], scoring=scoring, cv=cv)
        score = sum(scores) / len(scores)

        if verbose:
            print(f'Estimator: {type(regr).__name__}, Score: {score}')

        return score, param_grid

    def save_best_model(self, estimator, hrchy_code: str):
        f = open(os.path.join('..', '..', 'simulation', 'best_models',
                 self.data_version + '_' + self.division + '_' + str(self.hrchy_lvl) +
                              '_' + hrchy_code + '.pickle'), 'wb')
        pickle.dump(estimator, f)
        f.close()

    def save_scaler(self, scaler, hrchy_code: str):
        f = open(os.path.join('..', '..', 'simulation', 'scaler',
                 self.data_version + '_' + self.division + '_' + str(self.hrchy_lvl) +
                              '_' + hrchy_code + '.pickle'), 'wb')
        pickle.dump(scaler, f)
        f.close()

    def split_data(self, data):
        y_train = data[self.target_col]
        x_train = data.drop(columns=[self.target_col])

        return {'x_train': x_train, 'y_train': y_train}

    @staticmethod
    def scaling(data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        return scaler, data_scaled