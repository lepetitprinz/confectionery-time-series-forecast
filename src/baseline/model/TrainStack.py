import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.model.Algorithm import Algorithm

import os
import ast
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Sequence
from itertools import product
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


warnings.filterwarnings('ignore')


class Train(object):
    estimators_ts = {
        'ar': Algorithm.ar,
        'arima': Algorithm.arima,
        'hw': Algorithm.hw,
        'var': Algorithm.var,
        'varmax': Algorithm.varmax,
        'sarima': Algorithm.sarimax
    }
    estimators_ml = {
        'rf': RandomForestRegressor,    # Random Forest Regression
        'gb': GradientBoostingRegressor,    # Gradient Boosting Regression
        'et': ExtraTreesRegressor           # Extreme Random Tree Regression
    }

    def __init__(self, division: str, data_vrsn_cd: str, common: dict, hrchy: dict,
                 data_cfg: dict, exec_cfg: dict, mst_info: dict, exg_list: list, path_root: str):
        """
        :param division: Division (SELL-IN/SELl-OUT)
        :param data_vrsn_cd: Data version code
        :param common: Common information
        :param hrchy: Hierarchy information
        :param data_cfg: Data configuration
        :param exec_cfg: Execution configuration
        :param mst_info: Several master information
        :param exg_list: Exogenous variable list
        """
        # Class instance attribute
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Data instance attribute
        self.path_root = path_root
        self.common = common    # Common information
        self.data_cfg = data_cfg    # Data configuration
        self.exec_cfg = exec_cfg    # Execute configuration
        self.division = division    # SELL-IN / SELL-OUT
        self.data_vrsn_cd = data_vrsn_cd    # Data version code
        self.target_col = common['target_col']    # Target column

        # self.exo_col_list = exg_list + ['discount']    # Exogenous features
        self.exo_col_list = exg_list + common['exg_fixed'].split(',')
        self.cust_grp = mst_info['cust_grp']    # Customer group master
        self.item_mst = mst_info['item_mst']    # Item master

        # Data Level instance attribute
        self.cnt = 0    # Data level count
        self.hrchy = hrchy    # Hierarchy information

        # Time Series algorithm instance attribute
        self.model_info = mst_info['model_mst']    # Algorithm master
        self.param_grid = mst_info['param_grid']    # Hyper-parameter master
        self.model_candidates = list(self.model_info.keys())    # Model candidates list
        # self.param_grid_list = {}  # Hyper-parameter
        self.ts_param_grids = config.PARAM_GRIDS_FCST    # Hyper-parameter

        # Training instance attribute
        self.decimal_point = 3
        self.fixed_n_test = 4
        self.err_val = float(10 ** 5 - 1)    # set error values or clip outlier values
        self.validation_method = 'train_test'    # Train-test / Walk-forward
        self.grid_search_yn = exec_cfg['grid_search_yn']    # Execute grid search or not
        self.best_params_cnt = defaultdict(lambda: defaultdict(int))

        # Window Generator instance attribute
        self.window_method = 'expand'    # slice / expand
        self.hist_date = pd.date_range(
            start=self.data_cfg['date']['history']['from'],
            end=self.data_cfg['date']['history']['to'],
            freq='W'
        )
        self.hist_width = len(self.hist_date)
        self.val_width = int(common['week_eval'])
        self.pred_width = int(common['week_pred'])
        self.shift_width = 4
        self.end_width = self.hist_width + self.pred_width
        self.start_index = self.hist_width % self.shift_width
        self.window_cnt = (self.hist_width - self.start_index) / self.shift_width
        self.input_length = self.end_width - self.start_index
        self.train_length = self.hist_width - self.start_index - self.shift_width
        self.target_start_idx = self.start_index + self.shift_width

        # Stacking ensemble algorithm instance attribute
        self.stack_train_rate = 0.7
        self.stack_val_rate = 1 - self.stack_train_rate
        self.stack_test_size = self.pred_width
        self.stack_scoring = 'neg_root_mean_squared_error'
        self.stack_grid_search_yn = exec_cfg['stack_grid_search_yn']
        self.stack_grid_search_space = util.conv_json_to_dict(
            path=os.path.join(self.path_root, 'config', 'grid_search_space_stack.json')
        )
        self.stack_cv_fold = 5
        self.hyper_param_apply_option = 'each'
        self.model_param_by_data_lvl_map = {}
        self.grid_search_best_param_cnt = defaultdict(lambda: defaultdict(int))

        # After processing instance attribute
        self.fill_na_chk_list = ['cust_grp_nm', 'item_attr03_nm', 'item_attr04_nm', 'item_nm']
        self.rm_special_char_list = ['item_attr03_nm', 'item_attr04_nm', 'item_nm']

    def train(self, df) -> dict:
        if self.exec_cfg['stack_grid_search_yn']:
            scores = util.hrchy_recursion_add_key(
                hrchy_lvl=self.hrchy['lvl']['total'] - 1,
                fn=self.evaluation_model_with_hrchy,
                df=df
            )

        else:
            scores = util.hrchy_recursion_score(
                hrchy_lvl=self.hrchy['lvl']['total'] - 1,
                fn=self.evaluation_model,
                df=df
            )

        return scores

    def evaluation_model(self, df) -> tuple:
        # Print training progress
        self.cnt += 1
        if (self.cnt % 1000 == 0) or (self.cnt == self.hrchy['cnt']):
            print(f"Progress: ({self.cnt} / {self.hrchy['cnt']})")

        score_ts = self.train_time_series(df=df)
        score_ml, stack_data = self.train_stack_ensemble(df=df)
        # scores = self.concat_score(score_ts=score_ts, score_ml=score_ml)
        scores = score_ts + score_ml

        return scores, stack_data

    def evaluation_model_with_hrchy(self, hrchy, df) -> tuple:
        # Print training progress
        self.cnt += 1
        if (self.cnt % 1000 == 0) or (self.cnt == self.hrchy['cnt']):
            print(f"Progress: ({self.cnt} / {self.hrchy['cnt']})")

        score_ts = self.train_time_series(df=df)
        score_ml, stack_data = self.train_stack_ensemble(df=df, hrchy=hrchy)
        # scores = self.concat_score(score_ts=score_ts, score_ml=score_ml)
        scores = score_ts + score_ml

        return scores, stack_data

    def concat_score(self, score_ts, score_ml):
        if not self.exec_cfg['grid_search_yn']:
            score_ts = [[score[0], score[1]] for score in score_ts]
            scores = score_ts + score_ml
        else:
            scores = score_ts + score_ml

        # Sort scores
        scores = sorted(scores, key=lambda x: x[1])

        return scores

    def train_stack_ensemble(self, df, hrchy=None):
        # Generate machine learning input
        data_input = self.generate_input(data=df)

        # Fill empty data
        target = self.fill_empty_date(data=df[self.target_col])

        # Add target data
        data = self.add_target_data(input_data=data_input, target_data=target)

        # Make machine learning data
        data_fit = self.make_ml_data(data=data, kind='fit')
        data_pred = self.make_ml_data(data=data, kind='pred')

        # Evaluation
        scores = []
        for estimator_nm, estimator in self.estimators_ml.items():
            if self.stack_grid_search_yn:
                score, params = self.stack_grid_search_cv(
                    data=data_fit,
                    estimator=estimator(),
                    param_grid=self.stack_grid_search_space[estimator_nm],
                    scoring=self.stack_scoring,
                    cv=self.stack_cv_fold,
                )

            else:
                params = {}
                score = self.validation_ml_model(
                    data=data_fit,
                    estimator=estimator,
                    # params=self.ml_param_list[estimator_nm]
                )
            scores.append([estimator_nm, score, [], params])

        return scores, data_pred

    def fill_empty_date(self, data: pd.Series):
        empty_date = set(self.hist_date) - set(data.index)
        if len(empty_date) > 0:
            avg_qty = round(data.mean(), self.decimal_point)
            empty_df = pd.DataFrame([avg_qty] * len(empty_date), index=list(empty_date))
            data = pd.concat([data, empty_df], axis=0)
            data = data.sort_index().squeeze()

        return data

    def stack_grid_search_cv(self, data, estimator, param_grid: dict, scoring: str, cv: int):
        gsc = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        result = gsc.fit(data['train']['x'], data['train']['y'])

        return round(abs(result.best_score_), self.decimal_point), result.best_params_

    def make_ml_data(self, data, kind: str):
        # Split the data
        data_split = self.split_ml_fit_data(data=data, kind=kind)

        # Scale the data
        data_scaled = self.scaling(data=data_split, kind=kind)

        return data_scaled

    def validation_ml_model(self, data, estimator, params={}):
        est = estimator()
        est.set_params(**params)

        est.fit(data['train']['x'], data['train']['y'])
        yhat = est.predict(data['val']['x'])

        score = round(mean_squared_error(data['val']['y'], yhat, squared=False), self.decimal_point)

        return score

    def add_target_data(self, input_data, target_data):
        # target_data = target_data.reset_index(drop=True)
        target_sliced = target_data.iloc[self.target_start_idx:].tolist()
        target_sliced = pd.Series(target_sliced + [0] * self.pred_width)
        target_sliced = target_sliced.rename(self.target_col)
        concat_data = pd.concat([input_data, target_sliced], axis=1)
        concat_data = concat_data.fillna(0)

        return concat_data

    def split_ml_fit_data(self, data: pd.DataFrame, kind: str) -> dict:
        data_split = {}
        if kind == 'fit':
            train_data = data.iloc[self.start_index: self.train_length - self.val_width].copy()
            val_data = data.iloc[self.train_length - self.val_width: self.train_length].copy()
            test_data = data.iloc[self.train_length:].copy()

            data_split = {
                'train': {
                    'x': train_data.drop(columns=self.target_col).copy(),
                    'y': train_data[self.target_col]
                },
                'val': {
                    'x': val_data.drop(columns=self.target_col).copy(),
                    'y': val_data[self.target_col]
                },
                'test': {
                    'x': test_data.drop(columns=self.target_col).copy()
                }
            }
        elif kind == 'pred':
            train_data = data.iloc[self.start_index: self.train_length].copy()
            test_data = data.iloc[self.train_length:].copy()

            data_split = {
                'train': {
                    'x': train_data.drop(columns=self.target_col).copy(),
                    'y': train_data[self.target_col]
                },
                'test': {
                    'x': test_data.drop(columns=self.target_col).copy()
                }
            }

        return data_split

    def split_ml_train_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_data = data.iloc[self.train_length:].copy()
        data_sliced = data.iloc[:self.train_length].copy()
        train_data = data_sliced.iloc[:int(self.train_length * self.stack_train_rate)].copy()
        val_data = data_sliced.iloc[int(self.train_length * self.stack_train_rate):].copy()

        return train_data, val_data, test_data

    def generate_input(self, data: pd.DataFrame):
        # Set features by models (univ/multi)
        feature_by_variable = self.select_feature_by_variable(df=data)

        input_by_model = {}
        for model in self.model_candidates:
            # Classify dataset by variables
            data = feature_by_variable[self.model_info[model]['variate']]

            # Generate the machine learning input
            input_by_model[model] = self.predict_on_window_list(model=model, data=data)

        input_df = pd.DataFrame(input_by_model)

        return input_df

    def predict_on_window_list(self, model: str, data: list):
        data_window = self.split_window(data=data)

        ml_input = []
        for i, window in enumerate(data_window):
            if i == (self.window_cnt - 1):
                pred_width = self.pred_width
            else:
                pred_width = self.shift_width

            prediction = [0] * pred_width
            if self.model_info[model]['variate'] == 'multi':
                window = self.split_variable(data=window)

            # Predict
            try:
                prediction = self.predict(model=model, data=window, pred_width=pred_width)
                prediction = np.round(np.clip(prediction, 0, self.err_val).tolist(), self.decimal_point)
            except ValueError:
                continue

            ml_input.extend(prediction)

        return ml_input

    def predict(self, model: str, data, pred_width):
        try:
            prediction = self.estimators_ts[model](
                history=data,
                cfg=self.param_grid[model],
                pred_step=pred_width
                )
            # Clip results & Round results
            prediction = np.round(np.clip(prediction, 0, self.err_val).tolist(), self.decimal_point)

        except ValueError:
            value = 0
            if self.model_info[model]['variate'] == 'univ':
                value = round(data.mean(), self.decimal_point)
            elif self.model_info[model]['variate'] == 'multi':
                value = round(data['endog'].mean(), self.decimal_point)
            prediction = [value] * pred_width

        return prediction

    def split_variable(self, data) -> dict:
        data = {
            'endog': data[self.target_col].values.ravel(),
            'exog': data[self.exo_col_list].values
        }

        return data

    def fill_na_date(self, data):
        avg = round(data.mean(), self.decimal_point)
        na_date = list(set(self.hist_date) - set(data.index))
        if isinstance(data, pd.Series):
            na_df = pd.Series(avg, index=na_date)
        elif isinstance(data, pd.DataFrame):
            temp = avg.to_frame().transpose()
            temp = temp.append([temp] * (len(na_date) - 1), ignore_index=True)
            na_df = pd.DataFrame(
                temp.values,
                index=na_date,
                columns=temp.columns)
        data = data.append(na_df).sort_index()

        return data

    def split_window(self, data):
        if len(data) < self.hist_width:
            data = self.fill_na_date(data)

        window_list = []
        if self.window_method == 'expand':
            for i in range(self.start_index + self.shift_width, self.hist_width + 1, self.shift_width):
                window_list.append(data.iloc[self.start_index:i])
        elif self.window_method == 'slice':
            for i in range(self.start_index, self.hist_width + 1, self.shift_width):
                window_list.append(data.iloc[i:i + self.shift_width])

        return window_list

    # Training: Time Series
    def train_time_series(self, df) -> List[List[np.array]]:
        # Set features by models (univ/multi)
        feature_by_variable = self.select_feature_by_variable(df=df)

        scores = []
        for model in self.model_candidates:
            # Validation
            score, diff, params = self.validation(
                data=feature_by_variable[self.model_info[model]['variate']],
                model=model
            )
            scores.append([model, score, diff, params])

        if self.exec_cfg['voting_yn']:
            score = self.voting(models=scores)
            scores.append(['voting', score, [], {}])

        # scores = sorted(scores, key=lambda x: x[1])

        return scores

    def voting(self, models: list) -> float:
        score = self.err_val
        try:
            score = np.sqrt(np.mean((np.array([score[2] for score in models]).sum(axis=0) / len(models))**2, axis=0))
            if score > self.err_val:
                score = self.err_val
        except ValueError:
            pass

        return round(score, self.decimal_point)

    # Split univariate / multivariate features
    def select_feature_by_variable(self, df: pd.DataFrame) -> dict:
        feature_by_variable = None
        try:
            feature_by_variable = {'univ': df[self.target_col],    # Univariate columns
                                   'multi': df[self.exo_col_list + [self.target_col]]}    # Multivariate columns
        except ValueError:
            print("Data dose not have some columns")

        return feature_by_variable

    # Validation
    def validation(self, data, model: str) -> Tuple[float, Sequence, dict]:
        # Train / Test Split method
        if self.validation_method == 'train_test':
            score = self.train_test_validation(data=data, model=model)

        # Walk-forward method
        elif self.validation_method == 'walk_forward':
            score = self.walk_fwd_validation(data=data, model=model)

        else:
            raise ValueError

        return score

    def train_test_validation(self, model: str, data) -> Tuple[float, Sequence, dict]:
        # Set test length
        n_test = ast.literal_eval(self.model_info[model]['label_width'])

        # Split train & test dataset
        data_train, data_test = self.split_train_test(data=data, model=model, n_test=n_test)

        best_params = {}
        if self.grid_search_yn:
            # Grid Search
            score, diff, best_params = self.grid_search(
                model=model,
                train=data_train,
                test=data_test,
                n_test=n_test
            )

        else:
            # Evaluation
            score, diff = self.evaluation(
                model=model,
                params=self.param_grid[model],
                train=data_train,
                test=data_test,
                n_test=n_test
            )

        return score, diff, best_params

    def split_train_test(self, data: pd.DataFrame, model: str, n_test: int) -> Tuple[dict, dict]:
        data_length = len(data)

        if data_length - n_test >= n_test:    # if training period bigger than prediction
            data_train = data.iloc[: data_length - n_test]
            data_test = data.iloc[data_length - n_test:]

        elif data_length > self.fixed_n_test:    # if data period bigger than fixed period
            data_train = data.iloc[: data_length - self.fixed_n_test]
            data_test = data.iloc[data_length - self.fixed_n_test:]

        else:
            data_train = data.iloc[: data_length - 1]
            data_test = data.iloc[data_length - 1:]

        if self.model_info[model]['variate'] == 'multi':
            x_train = data_train[self.exo_col_list].values
            x_test = data_test[self.exo_col_list].values

            data_train = {
                'endog': data_train[self.target_col].values.ravel(),    # Target variable
                'exog': x_train    # Input variable
            }
            data_test = {
                'endog': data_test[self.target_col].values,    # Target variable
                'exog': x_test    # Input variable
            }

        return data_train, data_test

    # Calculate accuracy
    def calc_accuracy(self, test, pred) -> float:
        pred = np.where(pred < 0, 0, pred)    # change minus values to zero
        arr_acc = np.array([test, pred]).T
        arr_acc_marked = arr_acc[arr_acc[:, 0] != 0]

        if len(arr_acc_marked) != 0:
            acc = np.average(arr_acc_marked[:, 1] / arr_acc_marked[:, 0])
            acc = round(acc, self.decimal_point)
        else:
            # acc = np.nan
            acc = self.err_val

        return acc

    # evaluation
    def evaluation(self, model, params, train, test, n_test) -> Tuple[float, Sequence]:
        # get the length of train dataset
        if self.model_info[model]['variate'] == 'univ':
            len_train = len(train)
            len_test = len(test)
        else:
            len_train = len(train['endog'])
            len_test = len(test['endog'])

        score = self.err_val
        diff = [self.err_val] * len_test
        if len_train >= self.fixed_n_test:   # Evaluate if data length is bigger than minimum threshold
            try:
                yhat = self.estimators_ts[model](
                    history=train,    # Train dataset
                    cfg=params,       # Hyper-parameter
                    pred_step=n_test  # Prediction range
                )
                if yhat is not None:
                    if len_test < n_test:
                        yhat = yhat[:len_test]
                    if self.model_info[model]['variate'] == 'univ':
                        score = round(mean_squared_error(test, yhat, squared=False), self.decimal_point)
                        diff = test - yhat
                        diff = diff.values
                        # acc = round(self.calc_accuracy(test=test, pred=yhat), self.decimal_point)

                    elif self.model_info[model]['variate'] == 'multi':
                        score = round(mean_squared_error(test['endog'], yhat, squared=False), self.decimal_point)
                        diff = test['endog'] - yhat
                        # acc = round(self.calc_accuracy(test=test['endog'], pred=yhat), self.decimal_point)

                    # Clip error values
                    if score > self.err_val:
                        score = self.err_val

            except ValueError:
                pass

        return score, diff

    def grid_search(self, model, train, test, n_test) -> Tuple[float, Sequence, dict]:
        # get hyper-parameter grid for current algorithm
        param_grid_list = self.get_param_list(model=model)

        err_list = []
        for params in param_grid_list:
            err, diff = self.evaluation(
                model=model,
                params=params,
                train=train,
                test=test,
                n_test=n_test
            )
            err_list.append((err, diff, params))

        err_list = sorted(err_list, key=lambda x: x[0])    # Sort result based on error score
        best_result = err_list[0]    # Get the best result of grid search

        return best_result

    def walk_fwd_validation(self, model: str, data) -> np.array:
        """
        :param model: Statistical model
        :param data: time series data
        :return:
        """
        # split dataset
        dataset = self.window_generator(df=data, model=model)

        # evaluation
        n_test = ast.literal_eval(self.model_info[model]['label_width'])    # Change data type
        predictions = []
        for train, test in dataset:
            yhat = self.estimators_ts[model](history=train, cfg=self.param_grid[model], pred_step=n_test)
            yhat = np.nan_to_num(yhat)
            err = mean_squared_error(test, yhat, squared=False)
            predictions.append(err)

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    @staticmethod
    def scaling(data: dict, kind: str) -> dict:
        # Apply Min-Max Scaling
        scaler = MinMaxScaler()

        x_train_scaled = scaler.fit_transform(data['train']['x'])
        x_test_scaled = scaler.transform(data['test']['x'])

        data['train']['x'] = x_train_scaled
        data['test']['x'] = x_test_scaled

        if kind == 'fit':
            x_val_scaled = scaler.transform(data['val']['x'])
            data['val']['x'] = x_val_scaled

        return data

    def make_best_params_data(self, model: str, params: dict) -> tuple:
        model = model.upper()    # Convert name to uppercase
        data, info = [], []
        for key, val in params.items():
            data.append([
                self.common['project_cd'],
                model,
                key.upper(),
                str(val)
            ])
            info.append({
                'project_cd': self.common['project_cd'],
                'stat_cd': model,
                'option_cd': key.upper()
            })
        param_df = pd.DataFrame(data, columns=['PROJECT_CD', 'STAT_CD', 'OPTION_CD', 'OPTION_VAL'])

        return param_df, info

    def get_param_list(self, model) -> List[dict]:
        param_grids = self.ts_param_grids[model]    # Hyper-parameter list
        params = list(param_grids.keys())    # Hyper-parameter options
        values = param_grids.values()    # Hyper-parameter values
        values_combine_list = list(product(*values))

        values_combine_map_list = []
        for values_combine in values_combine_list:
            values_combine_map_list.append(dict(zip(params, values_combine)))

        return values_combine_map_list

    # Save best hyper-parameter based on counting
    def save_best_params_ts(self, scores) -> None:
        # Count best params for each data level
        util.hrchy_recursion(
            hrchy_lvl=self.hrchy['lvl']['total'] - 1,
            fn=self.count_best_params,
            df=scores
        )

        for model, count in self.best_params_cnt.items():
            params = [(val, key) for key, val in count.items()]
            params = sorted(params, key=lambda x: x[0], reverse=True)
            best_params = eval(params[0][1])
            best_params, params_info_list = self.make_best_params_data(model=model, params=best_params)

            for params_info in params_info_list:
                self.io.delete_from_db(sql=self.sql_conf.del_hyper_params(**params_info))
            self.io.insert_to_db(df=best_params, tb_name='M4S_I103011')

    def save_best_params_stack(self, scores) -> None:
        if self.hyper_param_apply_option == 'best':
            self.save_most_cnt_params(scores=scores)

        elif self.hyper_param_apply_option == 'each':
            self.save_each_params(scores=scores)

    # Save the most counted hyper-parameter set
    def save_most_cnt_params(self, scores) -> None:
        # Count best params for each data level
        util.hrchy_recursion(
            hrchy_lvl=self.hrchy['lvl']['total'] - 1,
            fn=self.count_best_params,
            df=scores
        )

        for model, count in self.grid_search_best_param_cnt.items():
            params = [(val, key) for key, val in count.items()]
            params = sorted(params, key=lambda x: x[0], reverse=True)
            best_params = eval(params[0][1])
            best_params, params_info_list = self.make_best_params_data(model=model, params=best_params)

            for params_info in params_info_list:
                self.io.delete_from_db(sql=self.sql_conf.del_hyper_params(**params_info))
            self.io.insert_to_db(df=best_params, tb_name='M4S_I103011')

    def save_each_params(self, scores) -> None:
        # Count best params for each data level
        util.hrchy_recursion_with_key(
            hrchy_lvl=self.hrchy['lvl']['total'] - 1,
            fn=self.make_model_param_map,
            df=scores
        )
        file_path = os.path.join(
            self.path_root, 'parameter', 'data_lvl_model_param_' + self.division + '_' +
                                         self.hrchy['key'][:-1] + '_' + str(self.n_test) + '.json'
        )
        self.io.save_object(data=self.model_param_by_data_lvl_map, data_type='json', file_path=file_path)

    # Make the mapping dictionary
    def make_model_param_map(self, hrchy, data):
        model_param_map = {}
        for eval_result in data:
            if eval_result[0] != 'voting':
                model, _, _, params = eval_result
                model_param_map[model] = params

        self.model_param_by_data_lvl_map['_'.join(hrchy)] = model_param_map


    # Count best hyper-parameters
    def count_best_params(self, data) -> None:
        for algorithm in data['score']:
            if algorithm[0] != 'voting':
                model, score, diff, params = algorithm
                self.best_params_cnt[model][str(params)] += 1

    def make_score_result(self, data: dict, hrchy_key: str, fn) -> Tuple[pd.DataFrame, dict]:
        hrchy_tot_lvl = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1
        result = util.hrchy_recursion_extend_key(hrchy_lvl=hrchy_tot_lvl, fn=fn, df=data)

        # Convert to dataframe
        result = pd.DataFrame(result)
        cols = self.hrchy['apply'] + ['stat_cd', 'rmse', 'diff']
        result.columns = cols
        result = result.drop(columns=['diff'])

        # Add information
        result['project_cd'] = self.common['project_cd']    # Project code
        result['division_cd'] = self.division
        result['data_vrsn_cd'] = self.data_vrsn_cd
        result['create_user_cd'] = 'SYSTEM'

        if hrchy_key[:2] == 'C1':    # if hierarchy contains SP1 (Customer group)
            if hrchy_key[3:5] == 'P5':    # if hierarchy contains SKU code
                result['fkey'] = hrchy_key + result['cust_grp_cd'] + '-' + result['sku_cd']
            else:
                key = self.hrchy['apply'][-1]
                result['fkey'] = hrchy_key + result['cust_grp_cd'] + '-' + result[key]
        else:
            key = self.hrchy['apply'][-1]
            result['fkey'] = hrchy_key + result[key]

        result['rmse'] = result['rmse'].fillna(0)
        # result['accuracy'] = result['accuracy'].where(pd.notnull(result['accuracy']), None)

        # Merge information
        # 1.Item code & name
        if self.hrchy['lvl']['item'] > 0:
            result = pd.merge(result,
                              self.item_mst[config.COL_ITEM[: 2 * self.hrchy['lvl']['item']]].drop_duplicates(),
                              on=self.hrchy['list']['item'][:self.hrchy['lvl']['item']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # 2.SP1 code & name
        if self.hrchy['lvl']['cust'] > 0:
            result = pd.merge(result,
                              self.cust_grp[config.COL_CUST[: 2 * self.hrchy['lvl']['cust']]].drop_duplicates(),
                              on=self.hrchy['list']['cust'][:self.hrchy['lvl']['cust']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
            # result = result.fillna('-')

        # Fill null values
        result = util.fill_na(data=result, chk_list=self.fill_na_chk_list)

        # Rename columns
        result = result.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)
        result = result.rename(columns=config.HRCHY_SKU_TO_DB_SKU_MAP)

        # Remove Special Character
        for col in self.rm_special_char_list:
            if col in list(result.columns):
                result = util.remove_special_character(data=result, feature=col)

        # set score information used to delete previous results
        score_info = {
            'project_cd': self.common['project_cd'],
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'fkey': hrchy_key[:-1]
        }

        return result, score_info

    def make_ml_data_map(self, data, fn):
        hrchy_tot_lvl = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1
        result = util.hrchy_recursion_extend_key(hrchy_lvl=hrchy_tot_lvl, fn=fn, df=data)
        hrchy_data_map = dict(result)

        return hrchy_data_map

    @staticmethod
    def make_hrchy_data_dict(hrchy: list, data):
        hrchy_key = '_'.join(hrchy)

        return [[hrchy_key, data['data']]]

    @staticmethod
    # Save all of scores to dataframe
    def score_to_df(hrchy: list, data) -> List[list]:
        result = []
        for algorithm, err, _, _ in data['score']:
            # result.append(hrchy + [algorithm.upper(), score])
            result.append(hrchy + [algorithm.upper(), err, 0])

        return result

    @staticmethod
    # Save best scores to dataframe
    def make_best_score_df(hrchy: list, data) -> list:
        result = []
        for algorithm, err, _, _ in data['score']:
            # result.append(hrchy + [algorithm.upper(), score])
            result.append(hrchy + [algorithm.upper(), err, 0])

        result = sorted(result, key=lambda x: x[-2])

        return [result[0]]

    # Make the sliding window data
    def window_generator(self, df, model: str) -> List[Tuple]:
        data_length = len(df)
        input_width = int(self.model_info[model]['input_width'])
        label_width = int(self.model_info[model]['label_width'])
        data_input = None
        data_target = None
        dataset = []
        for i in range(data_length - input_width - label_width + 1):
            # Univariate variable
            if self.model_info[model]['variate'] == 'univ':
                data_input = df.iloc[i: i + input_width]
                data_target = df.iloc[i + input_width: i + input_width + label_width]
            # Multivariate variable
            elif self.model_info[model]['variate'] == 'multi':
                data_input = df.iloc[i: i + input_width, :]
                data_target = df.iloc[i + input_width: i + input_width + label_width, :]

            dataset.append((data_input, data_target))

        return dataset
