import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import ast
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import mean_squared_error

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

    def __init__(self, division: str, mst_info: dict, date: dict, data_vrsn_cd: str,
                 exg_list: list, hrchy_lvl_dict: dict, hrchy_dict: dict, common: dict):
        # Data Configuration
        self.date = date
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
        self.hrchy_lvl_dict = hrchy_lvl_dict
        self.hrchy_tot_lvl = hrchy_lvl_dict['cust_lvl'] + hrchy_lvl_dict['item_lvl'] - 1
        self.hrchy_cust = hrchy_dict['hrchy_cust']
        self.hrchy_item = hrchy_dict['hrchy_item']
        self.hrchy = self.hrchy_cust[:hrchy_lvl_dict['cust_lvl']] + self.hrchy_item[:hrchy_lvl_dict['item_lvl']]

        # Algorithm Configuration
        self.param_grid = mst_info['param_grid']
        self.model_info = mst_info['model_mst']
        self.model_candidates = list(self.model_info.keys())

        # Training Configuration
        self.validation_method = config.VALIDATION_METHOD

    def train(self, df) -> dict:
        scores = util.hrchy_recursion(hrchy_lvl=self.hrchy_tot_lvl,
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
            print("Data dose not have some columns  `   ")

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
        result = util.hrchy_recursion_extend_key(hrchy_lvl=self.hrchy_tot_lvl,
                                                 fn=fn,
                                                 df=data)

        result = pd.DataFrame(result)
        cols = self.hrchy + ['stat_cd', 'rmse']
        result.columns = cols

        result['project_cd'] = self.common['project_cd']
        result['division_cd'] = self.division
        result['data_vrsn_cd'] = self.data_vrsn_cd
        result['create_user'] = 'SYSTEM'
        result['fkey'] = [hrchy_key + str(i+1).zfill(3) for i in range(len(result))]
        result['rmse'] = result['rmse'].fillna(0)

        # Merge information
        # Item Names
        if self.hrchy_lvl_dict['item_lvl'] > 0:
            result = pd.merge(result,
                              self.item_mst[config.COL_ITEM[: 2 * self.hrchy_lvl_dict['item_lvl']]].drop_duplicates(),
                              on=self.hrchy_item[:self.hrchy_lvl_dict['item_lvl']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # Customer Names
        if self.hrchy_lvl_dict['cust_lvl'] > 0:
            result = pd.merge(result,
                              self.cust_grp[config.COL_CUST[: 2 * self.hrchy_lvl_dict['cust_lvl']]].drop_duplicates(),
                              on=self.hrchy_cust[:self.hrchy_lvl_dict['cust_lvl']],
                              how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
            result = result.fillna('-')

        # Customer Names
        result = result.rename(columns=config.COL_RENAME1)
        result = result.rename(columns=config.COL_RENAME2)

        # set score_info
        score_info = {'project_cd': self.common['project_cd'],
                      'data_vrsn_cd': self.data_vrsn_cd,
                      'division_cd': self.division,
                      'fkey': hrchy_key[:-1]}

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
        data_train = None
        data_test = None
        data_length = len(data)
        n_test = ast.literal_eval(self.model_info[model]['label_width'])

        if self.model_info[model]['variate'] == 'univ':
            data_train = data.iloc[: data_length - n_test]
            data_test = data.iloc[data_length - n_test:]
        elif self.model_info[model]['variate'] == 'multi':
            data_train = data.iloc[: data_length - n_test, :]
            data_train = {'endog': data_train[self.target_col].values.ravel(),
                          'exog': data_train[self.exo_col_list].values}
            data_test = data.iloc[data_length - n_test:, :]
            data_test = {'endog': data_test[self.target_col],
                         'exog': data_test[self.exo_col_list].values}

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
