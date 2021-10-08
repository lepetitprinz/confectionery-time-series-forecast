import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import ast
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd


class Predict(object):
    def __init__(self, division: str, mst_info: dict, date: dict, data_vrsn_cd: str,
                 exg_list: list, hrchy_lvl_dict: dict, hrchy_dict: dict, common: dict):
        # Class Configuration
        self.algorithm = Algorithm()

        # Data Configuration
        self.date = date
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division    # SELL-IN / SELL-OUT
        self.target_col = common['target_col']    # Target features
        self.exo_col_list = exg_list + ['discount']    # Exogenous features
        self.cust_code = mst_info['cust_code']
        self.cust_grp = mst_info['cust_grp']
        self.item_mst = mst_info['item_mst']
        self.cal_mst = mst_info['cal_mst']

        # Data Level Configuration
        self.hrchy_lvl_dict = hrchy_lvl_dict
        self.hrchy_tot_lvl = hrchy_lvl_dict['cust_lvl'] + hrchy_lvl_dict['item_lvl'] - 1
        self.hrchy_cust = hrchy_dict['hrchy_cust']
        self.hrchy_item = hrchy_dict['hrchy_item']
        self.hrchy = self.hrchy_cust[:hrchy_lvl_dict['cust_lvl']] + self.hrchy_item[:hrchy_lvl_dict['item_lvl']]

        # Algorithms
        self.param_grid = mst_info['param_grid']
        self.model_info = mst_info['model_mst']
        self.cand_models = list(self.model_info.keys())
        self.model_fn = {'ar': self.algorithm.ar,
                         'arima': self.algorithm.arima,
                         'hw': self.algorithm.hw,
                         'var': self.algorithm.var,
                         'sarima': self.algorithm.sarimax}

    def forecast(self, df):
        prediction = util.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_tot_lvl,
                                                   fn=self.forecast_model,
                                                   df=df)

        return prediction

    def forecast_model(self, hrchy, df):
        feature_by_variable = self.select_feature_by_variable(df=df)

        models = []
        for model in self.cand_models:
            data = feature_by_variable[self.model_info[model]['variate']]
            data = self.split_variable(model=model, data=data)
            n_test = ast.literal_eval(self.model_info[model]['label_width'])
            try:
                prediction = self.model_fn[model](history=data,
                                                  cfg=self.param_grid[model],
                                                  pred_step=n_test)
            except ValueError:
                prediction = [0] * n_test
            models.append(hrchy + [model.upper(), prediction])

        return models

    def select_feature_by_variable(self, df: pd.DataFrame):
        feature_by_variable = {'univ': df[self.target_col],
                               'multi': df[self.exo_col_list + [self.target_col]]}

        return feature_by_variable

    def split_variable(self, model: str, data) -> np.array:
        if self.model_info[model]['variate'] == 'multi':
            data = {'endog': data[self.target_col].values.ravel(),
                    'exog': data[self.exo_col_list].values}

        return data

    def make_pred_result(self, df, hrchy_key: str):
        end_date = datetime.strptime(self.date['date_to'], '%Y%m%d')
        end_date += timedelta(days=1)
        # end_date += timedelta(weeks=15) - timedelta(days=6)   # Todo: Exception

        result_pred = []
        fkey = [hrchy_key + str(i+1).zfill(3) for i in range(len(df))]
        for i, pred in enumerate(df):
            for j, prediction in enumerate(pred[-1]):
                result_pred.append([fkey[i]] + pred[:-1] +
                                   [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), prediction])
                # results.append([fkey[i]] + pred[:-1] + [pred[-1].index[j], result])

        result_pred = pd.DataFrame(result_pred)
        cols = ['fkey'] + self.hrchy + ['stat_cd', 'yymmdd', 'result_sales']
        result_pred.columns = cols
        result_pred['project_cd'] = 'ENT001'
        result_pred['division_cd'] = self.division
        result_pred['data_vrsn_cd'] = self.data_vrsn_cd
        result_pred['create_user'] = 'SYSTEM'

        if self.hrchy_lvl_dict['item_lvl'] > 0:
            result_pred = pd.merge(result_pred,
                                   self.item_mst[config.COL_ITEM[: 2 * self.hrchy_lvl_dict['item_lvl']]].drop_duplicates(),
                                   on=self.hrchy_item[:self.hrchy_lvl_dict['item_lvl']],
                                   how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        if self.hrchy_lvl_dict['cust_lvl'] > 0:
            result_pred = pd.merge(result_pred,
                                   self.cust_grp[config.COL_CUST[: 2 * self.hrchy_lvl_dict['cust_lvl']]].drop_duplicates(),
                                   on=self.hrchy_cust[:self.hrchy_lvl_dict['cust_lvl']],
                                   how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

            result_pred = result_pred.fillna('-')

        result_pred = pd.merge(result_pred, self.cal_mst, on='yymmdd', how='left')

        # Rename columns
        result_pred = result_pred.rename(columns=config.COL_RENAME1)
        result_pred = result_pred.rename(columns=config.COL_RENAME2)

        # Prediction information
        pred_info = {'project_cd': 'ENT001',
                     'data_vrsn_cd': self.data_vrsn_cd,
                     'division_cd': self.division,
                     'fkey': hrchy_key[:-1]}

        return result_pred, pred_info

    # def lstm_predict(self, train: pd.DataFrame, units: int) -> np.array:
    #     # scaling
    #     scaler = MinMaxScaler()
    #     train_scaled = scaler.fit_transform(train)
    #     train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    #
    #     x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
    #                                                n_steps_out=self.n_test)
    #     n_features = x_train.shape[2]
    #
    #     # Build model
    #     model = Sequential()
    #     model.add(LSTM(units=units, activation='relu', return_sequences=True,
    #               input_shape=(self.n_test, n_features)))
    #     # model.add(LSTM(units=units, activation='relu'))
    #     model.add(Dense(n_features))
    #     model.compile(optimizer='adam', loss=self.root_mean_squared_error)
    #
    #     model.fit(x_train, y_train,
    #               epochs=self.epoch_best,
    #               batch_size=config.BATCH_SIZE,
    #               shuffle=False,
    #               verbose=0)
    #     test = x_train[-1]
    #     test = test.reshape(1, test.shape[0], test.shape[1])
    #
    #     predictions = model.predict(test, verbose=0)
    #     predictions = scaler.inverse_transform(predictions[0])
    #
    #     return predictions[:, 0]