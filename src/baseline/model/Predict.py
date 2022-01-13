import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import ast
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd


class Predict(object):
    estimators = {
        'ar': Algorithm.ar,
        'arima': Algorithm.arima,
        'hw': Algorithm.hw,
        'var': Algorithm.var,
        'varmax': Algorithm.varmax,
        'sarima': Algorithm.sarimax
    }

    def __init__(self, division: str, mst_info: dict, date: dict, data_vrsn_cd: str,
                 exg_list: list, hrchy: dict, common: dict, data_cfg: dict):
        # Class Configuration
        self.algorithm = Algorithm()

        # Data Configuration
        self.data_cfg = data_cfg
        self.date = date
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division    # SELL-IN / SELL-OUT
        self.target_col = common['target_col']    # Target features
        self.exo_col_list = exg_list + ['discount']    # Exogenous features
        self.cust_grp = mst_info['cust_grp']
        self.item_mst = mst_info['item_mst']
        self.cal_mst = mst_info['cal_mst']

        # Data Level Configuration
        self.cnt = 0
        self.hrchy = hrchy
        self.fixed_n_test = 4

        # Algorithms
        self.err_val = 0
        self.max_val = float(10 ** 5 - 1)
        self.param_grid = mst_info['param_grid']
        self.model_info = mst_info['model_mst']
        self.cand_models = list(self.model_info.keys())

        # After processing Configuration
        self.fill_na_chk_list = ['cust_grp_nm', 'item_attr03_nm', 'item_attr04_nm', 'item_nm']
        self.rm_special_char_list = ['item_attr03_nm', 'item_attr04_nm', 'item_nm']

    def forecast(self, df):
        hrchy_tot_lvl = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1
        prediction = util.hrchy_recursion_extend_key(
            hrchy_lvl=hrchy_tot_lvl,
            fn=self.forecast_model,
            df=df
        )

        return prediction

    def forecast_model(self, hrchy, df):
        # Show prediction progress
        self.cnt += 1
        if (self.cnt % 1000 == 0) or (self.cnt == self.hrchy['cnt']):
            print(f"Progress: ({self.cnt} / {self.hrchy['cnt']})")

        # Set features by models (univ/multi)
        feature_by_variable = self.select_feature_by_variable(df=df)

        models = []
        for model in self.cand_models:
            data = feature_by_variable[self.model_info[model]['variate']]
            data = self.split_variable(model=model, data=data)
            n_test = ast.literal_eval(self.model_info[model]['label_width'])

            if self.model_info[model]['variate'] == 'univ':
                length = len(data)
            else:
                length = len(data['endog'])
            if length > self.fixed_n_test:
                try:
                    prediction = self.estimators[model](
                        history=data,
                        cfg=self.param_grid[model],
                        pred_step=n_test
                    )

                    prediction = np.clip(prediction, 0, self.max_val).tolist()  # Clip results
                    prediction = np.round(prediction, 3)  # Round results

                except ValueError:
                    prediction = [self.err_val] * n_test
            else:
                prediction = [self.err_val] * n_test
            models.append(hrchy + [model.upper(), prediction])

        return models

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

        if self.hrchy['lvl']['item'] > 0:
            result_pred = pd.merge(
                result_pred,
                self.item_mst[config.COL_ITEM[: 2 * self.hrchy['lvl']['item']]].drop_duplicates(),
                on=self.hrchy['list']['item'][: self.hrchy['lvl']['item']],
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

        if self.hrchy['lvl']['cust'] > 0:
            result_pred = pd.merge(
                result_pred,
                self.cust_grp[config.COL_CUST[: 2 * self.hrchy['lvl']['cust']]].drop_duplicates(),
                on=self.hrchy['list']['cust'][: self.hrchy['lvl']['cust']],
                how='left',
                suffixes=('', '_DROP')
            ).filter(regex='^(?!.*_DROP)')

            result_pred = result_pred.fillna('-')

        calendar = self.cal_mst[['yymmdd', 'week']]

        # Merge calendar data
        result_pred = pd.merge(result_pred, calendar, on='yymmdd', how='left')

        # Fill na
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
