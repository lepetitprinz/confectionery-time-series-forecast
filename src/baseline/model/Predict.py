import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import ast
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd


class Predict(object):
    def __init__(self, division: str, mst_info: dict, date: dict,
                 hrchy: list, common: dict):
        # Class Configuration
        self.algorithm = Algorithm()

        # Data Configuration
        self.division = division    # SELL-IN / SELL-OUT
        self.target_col = common['target_col']     # Target features
        self.exo_col_list = ['discount']     # Exogenous features
        self.cust_mst = mst_info['cust_mst']
        self.item_mst = mst_info['item_mst']
        self.cal_mst = mst_info['cal_mst']
        self.date = date

        # Data Level Configuration
        self.hrchy = hrchy
        self.hrchy_level = len(hrchy) - 1

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
        prediction = util.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_level,
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
                    'exog': data[self.exo_col_list].values.ravel()}

        return data

    def make_pred_result(self, df, hrchy_key: str):
        end_date = datetime.strptime(self.date['date_to'], '%Y%m%d')

        results = []
        fkey = [hrchy_key + str(i+1).zfill(3) for i in range(len(df))]
        for i, pred in enumerate(df):
            for j, result in enumerate(pred[-1]):
                results.append([fkey[i]] + pred[:-1] +
                               [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), result])
                # results.append([fkey[i]] + pred[:-1] + [pred[-1].index[j], result])

        results = pd.DataFrame(results)
        cols = ['fkey'] + self.hrchy + ['stat_cd', 'yymmdd', 'result_sales']
        results.columns = cols
        results['project_cd'] = 'ENT001'
        results['division_cd'] = self.division
        results['data_vrsn_cd'] = self.date['date_from'] + '-' + self.date['date_to']
        results['create_user'] = 'SYSTEM'

        results = pd.merge(results, self.item_mst[config.COL_NAMES[: 2*len(self.hrchy)]].drop_duplicates(),
                           on=self.hrchy, how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        results = pd.merge(results, self.cal_mst, on='yymmdd', how='left')

        # Rename columns
        results = results.rename(columns=config.COL_RENAME1)
        results = results.rename(columns=config.COL_RENAME2)


        return results

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