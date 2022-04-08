import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import os
import ast
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


class Predict(object):
    estimators_ts = {
        'ar': Algorithm.ar,            # Autoregressive model
        'arima': Algorithm.arima,      # Arima model
        'hw': Algorithm.hw,            # Holt-winters model
        'var': Algorithm.var,          # Vector Autoregressive model
        'varmax': Algorithm.varmax,    # VARMAX model
        'sarima': Algorithm.sarimax    # SARIMAX model
    }

    estimators_ml = {
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor,
        'et': ExtraTreesRegressor
    }
    ml_param_name = {
        'each': 'data_lvl_ml_model_param_',
        'best': 'ml_model_param_'
    }

    def __init__(self, io, date: dict, division: str, data_vrsn_cd: str, common: dict, hrchy: dict, data_cfg: dict,
                 exec_cfg: dict, path_root: str,  mst_info: dict, exg_list: list, ml_data_map: dict):
        """
        :param date: Date information
        :param division: Division (SELL-IN/SELl-OUT)
        :param data_vrsn_cd: Data version code
        :param common: Common information
        :param hrchy: Hierarchy information
        :param data_cfg: Data configuration
        :param mst_info: Several master information
        :param exg_list: Exogenous variable list
        """
        # Class instance attribute
        self.io = io
        self.algorithm = Algorithm()

        # Configuration instance attribute
        self.data_cfg = data_cfg            # Data configuration
        self.exec_cfg = exec_cfg            # Execution configuration

        self.date = date                    # Date information
        self.path_root = path_root
        self.division = division            # SELL-IN / SELL-OUT
        self.data_vrsn_cd = data_vrsn_cd    # Data version code
        self.target_col = common['target_col']          # Target features
        self.exo_col_list = exg_list + common['exg_fixed'].split(',')

        # Data instance attribute
        self.cal_mst = mst_info['cal_mst']              # Calendar master
        self.cust_grp = mst_info['cust_grp']            # Customer group master
        self.item_mst = mst_info['item_mst']            # Item master

        # Data Level instance attribute
        self.cnt = 0          # Data level count
        self.hrchy = hrchy    # Hierarchy information

        # Algorithms instance attribute
        self.err_val = 0                            # Setting value for prediction error
        self.fixed_n_test = 4
        self.max_val = float(10 ** 5 - 1)           # Clipping value for outlier
        self.model_info = mst_info['model_mst']     # Algorithm master
        self.param_grid = mst_info['param_grid']    # Hyper-parameter master
        self.cand_models = list(self.model_info.keys())

        # Stacking algorithm instance attribute
        self.ml_data_map = ml_data_map
        self.ml_hyper_parameter = {}
        self.ml_hyper_param_apply_option = 'each'

        # Post processing instance attribute
        self.fill_na_chk_list = ['cust_grp_nm', 'item_attr03_nm', 'item_attr04_nm', 'item_nm']
        self.rm_special_char_list = ['item_attr03_nm', 'item_attr04_nm', 'item_nm']

        self.decimal_point = 3
        self.voting_opt = 'mean'

    def init(self):
        param_path = os.path.join(self.path_root, 'parameter')
        file_name = self.division + '_' + self.hrchy['key'][:-1] + '.json'
        path_each = os.path.join(param_path, self.ml_param_name['each'] + file_name)
        path_best = os.path.join(param_path, self.ml_param_name['best'] + file_name)

        self.ml_hyper_parameter['each'] = self.io.load_object(file_path=path_each, data_type='json')
        self.ml_hyper_parameter['best'] = self.io.load_object(file_path=path_best, data_type='json')

    def forecast(self, df):
        # Initialize parameters
        self.init()

        # Forecast
        predictions = util.hrchy_recursion_extend_key(
            hrchy_lvl=self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1,
            fn=self.forecast_model,
            df=df
        )

        return predictions

    def forecast_model(self, hrchy, df):
        # Show prediction progress
        self.show_progress()

        prediction_ts = self.forecast_time_series(hrchy=hrchy, data=df)
        prediction_ml = self.forecast_machine_learning(hrchy=hrchy)
        prediction = prediction_ts + prediction_ml

        return prediction

    def show_progress(self) -> None:
        # Show prediction progress
        self.cnt += 1
        if (self.cnt % 1000 == 0) or (self.cnt == self.hrchy['cnt']):
            print(f"Progress: ({self.cnt} / {self.hrchy['cnt']})")

    def forecast_machine_learning(self, hrchy):
        # get data
        data = self.ml_data_map['_'.join(hrchy)]

        # Get algorithm hyper-parameter
        params = self.get_hyper_parameter(hrchy=hrchy)

        predictions = []
        for estimator_nm, estimator in self.estimators_ml.items():
            prediction = self.predict_ml_model(
                data=data,
                estimator=estimator,
                params=params[estimator_nm],
                # params=self.ml_param_list[estimator_nm]
            )
            prediction = np.round(np.clip(prediction, 0, self.max_val).tolist(), self.decimal_point)
            predictions.append(hrchy + [estimator_nm.upper(), prediction])

        return predictions

    def get_hyper_parameter(self, hrchy: list):
        # Get algorithm hyper-parameter
        param_grid = {}
        if self.ml_hyper_param_apply_option == 'best':
            param_grid = self.ml_hyper_parameter['best']

        elif self.ml_hyper_param_apply_option == 'each':
            param_grid = self.ml_hyper_parameter['each'].get('_'.join(hrchy), self.ml_hyper_parameter['best'])

        return param_grid

    @staticmethod
    def predict_ml_model(data, estimator, params={}):
        est = estimator()
        est.set_params(**params)

        # Fit the model
        est.fit(data['train']['x'], data['train']['y'])
        yhat = est.predict(data['test']['x'])

        return yhat

    def forecast_time_series(self, hrchy, data):
        # Set features by models (univ/multi)
        feature_by_variable = self.select_feature_by_variable(df=data)

        predictions = []
        for model in self.cand_models:
            data = feature_by_variable[self.model_info[model]['variate']]
            data = self.split_variable(model=model, data=data)
            n_test = ast.literal_eval(self.model_info[model]['label_width'])

            if len(data) > self.fixed_n_test:
                try:
                    prediction = self.estimators_ts[model](
                        history=data,
                        cfg=self.param_grid[model],
                        pred_step=n_test
                    )
                    # Clip results & Round results
                    prediction = np.round(np.clip(prediction, 0, self.max_val).tolist(), self.decimal_point)

                except ValueError:
                    prediction = [self.err_val] * n_test
            else:
                prediction = [self.err_val] * n_test
            predictions.append(hrchy + [model.upper(), prediction])

        if self.exec_cfg['voting_yn']:
            prediction = self.ensemble_voting(models=predictions)
            predictions.append(hrchy + ['VOTING', prediction])

        return predictions

    def ensemble_voting(self, models: list):
        ensemble = None
        if self.voting_opt == 'mean':
            ensemble = np.array([pred[5] for pred in models]).mean(axis=0).round(self.decimal_point)

        return ensemble

    def select_feature_by_variable(self, df: pd.DataFrame):
        feature_by_variable = {'univ': df[self.target_col],
                               'multi': df[self.exo_col_list + [self.target_col]]}

        return feature_by_variable

    def split_variable(self, model: str, data) -> np.array:
        if self.model_info[model]['variate'] == 'multi':
            data = {
                'endog': data[self.target_col].values.ravel(),
                'exog': data[self.exo_col_list].values
            }

        return data

    def make_db_format_pred_all(self, df, hrchy_key: str):
        end_date = datetime.strptime(self.date['history']['to'], '%Y%m%d')
        end_date -= timedelta(days=6)    # Week start day

        result_pred = []
        lvl = self.hrchy['lvl']['item'] - 1
        for i, pred in enumerate(df):
            for j, prediction in enumerate(pred[-1]):

                # Add data level information
                if hrchy_key[:2] == 'C1':
                    if hrchy_key[3:5] == 'P5':
                        result_pred.append(
                            [hrchy_key + pred[0] + '-' + pred[5]] + pred[:-1] +
                            [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), prediction]
                        )
                    else:
                        result_pred.append(
                            [hrchy_key + pred[0] + '-' + pred[lvl+1]] + pred[:-1] +
                            [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), prediction]
                        )
                else:
                    result_pred.append(
                        [hrchy_key + pred[lvl+1]] + pred[:-1] +
                        [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), prediction]
                    )

        result_pred = pd.DataFrame(result_pred)
        cols = ['fkey'] + self.hrchy['apply'] + ['stat_cd', 'yymmdd', 'result_sales']
        result_pred.columns = cols
        result_pred['project_cd'] = 'ENT001'
        result_pred['division_cd'] = self.division
        result_pred['data_vrsn_cd'] = self.data_vrsn_cd
        result_pred['create_user_cd'] = 'SYSTEM'

        # Merge master information
        # 1.Item code & name
        if self.hrchy['lvl']['item'] > 0:
            result_pred = pd.merge(
                result_pred,
                self.item_mst[config.COL_ITEM[: 2 * self.hrchy['lvl']['item']]].drop_duplicates(),
                on=self.hrchy['list']['item'][: self.hrchy['lvl']['item']],
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

        # 2.SP1 code & name
        if self.hrchy['lvl']['cust'] > 0:
            result_pred = pd.merge(
                result_pred,
                self.cust_grp[config.COL_CUST[: 2 * self.hrchy['lvl']['cust']]].drop_duplicates(),
                on=self.hrchy['list']['cust'][: self.hrchy['lvl']['cust']],
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

            result_pred = result_pred.fillna('-')

        # 3.Calendar information
        calendar = self.cal_mst[['yymmdd', 'week']]
        result_pred = pd.merge(result_pred, calendar, on='yymmdd', how='left')

        # Fill null values
        result_pred = util.fill_na(data=result_pred, chk_list=self.fill_na_chk_list)

        # Rename columns
        result_pred = result_pred.rename(columns=config.HRCHY_CD_TO_DB_CD_MAP)
        result_pred = result_pred.rename(columns=config.HRCHY_SKU_TO_DB_SKU_MAP)

        # Remove Special Character
        for col in self.rm_special_char_list:
            if col in list(result_pred.columns):
                result_pred = util.remove_special_character(data=result_pred, feature=col)

        # Prediction information
        pred_info = {
            'project_cd': 'ENT001',
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'fkey': hrchy_key[:-1]
        }

        return result_pred, pred_info

    @staticmethod
    def make_db_format_pred_best(pred: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
        pred.columns = [col.lower() for col in list(pred.columns)]
        score.columns = [col.lower() for col in list(score.columns)]

        # Merge prediction and score dataset
        best = pd.merge(
            pred,
            score,
            how='inner',
            on=['data_vrsn_cd', 'division_cd', 'fkey', 'stat_cd'],
            suffixes=('', '_DROP')
        ).filter(regex='^(?!.*_DROP)')

        best = best.rename(columns={'create_user': 'create_user_cd'})
        best = best.drop(columns=['rmse', 'accuracy'], errors='ignore')

        return best
