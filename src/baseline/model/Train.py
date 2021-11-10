import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import ast
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Tensorflow library
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')


class Train(object):
    estimators = {'ar': Algorithm.ar,
                  'arima': Algorithm.arima,
                  'hw': Algorithm.hw,
                  'var': Algorithm.var,
                  'sarima': Algorithm.sarimax}

    def __init__(self, mst_info: dict, common: dict, division: str, data_vrsn_cd: str,
                 hrchy: dict, exg_list: list,  exec_cfg: dict):
        # Data Configuration
        self.exec_cfg = exec_cfg
        self.data_vrsn_cd = data_vrsn_cd
        self.common = common
        self.division = division    # SELL-IN / SELL-OUT
        self.target_col = common['target_col']    # Target column

        # self.exo_col_list = ['discount']     # Exogenous features
        self.exo_col_list = exg_list + ['discount']    # Exogenous features
        self.cust_code = mst_info['cust_code']
        self.cust_grp = mst_info['cust_grp']
        self.item_mst = mst_info['item_mst']

        # Data Level Configuration
        self.hrchy = hrchy

        # Algorithm Configuration
        self.param_grid = mst_info['param_grid']
        self.model_info = mst_info['model_mst']
        self.model_candidates = list(self.model_info.keys())

        # Training Configuration
        self.validation_method = config.VALIDATION_METHOD

    def train(self, df) -> dict:
        hrchy_tot_lvl = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1
        scores = util.hrchy_recursion(hrchy_lvl=hrchy_tot_lvl,
                                      fn=self.train_model,
                                      df=df)

        return scores

    def train_model(self, df) -> List[List[np.array]]:
        feature_by_variable = self.select_feature_by_variable(df=df)

        models = []
        for model in self.model_candidates:
            score = self.evaluation(data=feature_by_variable[self.model_info[model]['variate']], model=model)
            # Exception
            if score > 10 ** 20:
                score = float(10 ** 20)
            models.append([model, np.round(score, 3)])
        models = sorted(models, key=lambda x: x[1])

        return models

    def select_feature_by_variable(self, df: pd.DataFrame):
        feature_by_variable = None
        try:
            feature_by_variable = {'univ': df[self.target_col],
                                   'multi': df[self.exo_col_list + [self.target_col]]}
        except ValueError:
            print("Data dose not have some columns")

        return feature_by_variable

    def evaluation(self, data, model: str) -> float:
        # if len(data) > int(self.model_info[model]['input_width']):
        if self.validation_method == 'train_test':
            score = self.train_test_validation(data=data, model=model)

        elif self.validation_method == 'walk_forward':
            score = self.walk_fwd_validation(data=data, model=model)

        else:
            raise ValueError

        return score

    def make_score_result(self, data: dict, hrchy_key: str, fn):
        hrchy_tot_lvl = self.hrchy['lvl']['cust'] + self.hrchy['lvl']['item'] - 1
        result = util.hrchy_recursion_extend_key(hrchy_lvl=hrchy_tot_lvl,
                                                 fn=fn,
                                                 df=data)

        result = pd.DataFrame(result)
        cols = self.hrchy['apply'] + ['stat_cd', 'rmse']
        result.columns = cols

        result['project_cd'] = self.common['project_cd']
        result['division_cd'] = self.division
        result['data_vrsn_cd'] = self.data_vrsn_cd
        result['create_user_cd'] = 'SYSTEM'
        result['fkey'] = hrchy_key + result['cust_grp_cd'] + '-' + result['sku_cd']
        # result['fkey'] = [hrchy_key + str(i+1).zfill(5) for i in range(len(result))]
        result['rmse'] = result['rmse'].fillna(0)

        # Merge information
        # Item Names
        if self.hrchy['lvl']['item'] > 0:
            result = pd.merge(result,
                              self.item_mst[config.COL_ITEM[: 2 * self.hrchy['lvl']['item']]].drop_duplicates(),
                              on=self.hrchy['list']['item'][:self.hrchy['lvl']['item']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # Customer Names
        if self.hrchy['lvl']['cust'] > 0:
            result = pd.merge(result,
                              self.cust_grp[config.COL_CUST[: 2 * self.hrchy['lvl']['cust']]].drop_duplicates(),
                              on=self.hrchy['list']['cust'][:self.hrchy['lvl']['cust']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
            result = result.fillna('-')

        # Customer Names
        result = result.rename(columns=config.COL_RENAME1)
        result = result.rename(columns=config.COL_RENAME2)

        # set score_info
        score_info = {
            'project_cd': self.common['project_cd'],
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'fkey': hrchy_key[:-1]
        }

        return result, score_info

    @staticmethod
    def score_to_df(hrchy: list, data) -> List[list]:
        result = []
        for algorithm, score in data:
            result.append(hrchy + [algorithm.upper(), score])

        return result

    @staticmethod
    def best_score_to_df(hrchy: list, data) -> list:
        result = []
        for algorithm, score in data:
            result.append(hrchy + [algorithm.upper(), score])

        result = sorted(result, key=lambda x: x[2])

        return [result[0]]

    def train_test_validation(self, model: str, data) -> np.array:
        # split dataset
        n_test = ast.literal_eval(self.model_info[model]['label_width'])
        data_train, data_test = self.split_data(data=data, model=model, n_test=n_test)

        if self.model_info[model]['variate'] == 'multi':
            x_train = data_train[self.exo_col_list].values
            x_test = data_test[self.exo_col_list].values

            if self.exec_cfg['scaling_yn']:
                x_train, x_test = self.scaling(
                    train=data_train,
                    test=data_test
                )

            data_train = {
                'endog': data_train[self.target_col].values.ravel(),
                'exog': x_train
            }
            data_test = {
                'endog': data_test[self.target_col].values,
                'exog': x_test
            }

        # evaluation
        try:
            yhat = self.estimators[model](history=data_train, cfg=self.param_grid[model], pred_step=n_test)
            yhat = np.nan_to_num(yhat)

            err = 0
            if self.model_info[model]['variate'] == 'univ':
                err = mean_squared_error(data_test, yhat, squared=False)
            elif self.model_info[model]['variate'] == 'multi':
                err = mean_squared_error(data_test['endog'], yhat, squared=False)

        except ValueError:
            err = 10**10 - 1   # Not solvable problem

        return err

    def walk_fwd_validation(self, model: str, data) -> np.array:
        """
        :param model: Statistical model
        :param data: time series data
        :return:
        """
        # split dataset
        dataset = self.window_generator(df=data, model=model)

        # evaluation
        n_test = ast.literal_eval(self.model_info[model]['label_width'])
        predictions = []
        for train, test in dataset:
            yhat = self.estimators[model](history=train, cfg=self.param_grid[model], pred_step=n_test)
            yhat = np.nan_to_num(yhat)
            err = mean_squared_error(test, yhat, squared=False)
            predictions.append(err)

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def split_data(self, data: pd.DataFrame, model: str, n_test: int):
        data_length = len(data)

        data_train, data_test = None, None
        if self.model_info[model]['variate'] == 'univ':
            data_train = data.iloc[: data_length - n_test]
            data_test = data.iloc[data_length - n_test:]
        elif self.model_info[model]['variate'] == 'multi':
            data_train = data.iloc[: data_length - n_test, :]
            data_test = data.iloc[data_length - n_test:, :]

        return data_train, data_test

    def scaling(self, train, test) -> tuple:
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(train)
        x_test_scaled = scaler.transform(test)

        return x_train_scaled, x_test_scaled


    def window_generator(self, df, model: str) -> List[Tuple]:
        data_length = len(df)
        input_width = int(self.model_info[model]['input_width'])
        label_width = int(self.model_info[model]['label_width'])
        data_input = None
        data_target = None
        dataset = []
        for i in range(data_length - input_width - label_width + 1):
            if self.model_info[model]['variate'] == 'univ':
                data_input = df.iloc[i: i + input_width]
                data_target = df.iloc[i + input_width: i + input_width + label_width]
            elif self.model_info[model]['variate'] == 'multi':
                data_input = df.iloc[i: i + input_width, :]
                data_target = df.iloc[i + input_width: i + input_width + label_width, :]

            dataset.append((data_input, data_target))

        return dataset

    def train_test_split(self, data, model):
        train_rate = ast.literal_eval(self.common['train_rate'])
        data_length = len(data)
        if self.model_info[model]['variate'] == 'univ':
            return data[: int(data_length * train_rate)], data[int(data_length * train_rate):]

        elif self.model_info[model]['variate'] == 'multi':
            return data.iloc[:int(data_length * train_rate), :], data.iloc[int(data_length * train_rate):, :]

    # def grid_search(self, model: str, data, n_test: int, cfg_list: list):
    #     scores = [self.score_model(model=model, data=data, n_test=n_test,
    #                                cfg=cfg) for cfg in cfg_list]
    #     # remove empty results
    #     scores = [score for score in scores if score[1] != None]
    #     # sort configs by error, asc
    #     scores.sort(key=lambda tup: tup[2])
    #
    #     return scores

    # def lstm_train(self, train: pd.DataFrame, units: int) -> float:
    #     # scaling
    #     scaler = MinMaxScaler()
    #     train_scaled = scaler.fit_transform(train)
    #     train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    #
    #     x_train, y_train = DataPrep.split_sequence(df=train_scaled.values, n_steps_in=config.TIME_STEP,
    #                                                n_steps_out=self.n_test)
    #
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
    #     history = model.fit(x_train, y_train,
    #                         epochs=config.EPOCHS,
    #                         batch_size=config.BATCH_SIZE,
    #                         validation_split=1-config.TRAIN_RATE,
    #                         shuffle=False,
    #                         verbose=0)
    #     self.epoch_best = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    #
    #     rmse = min(history.history['val_loss'])
    #
    #     return rmse
