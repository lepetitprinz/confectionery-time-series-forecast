import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import warnings
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import mean_squared_error

# Tensorflow library
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')


class Train(object):
    def __init__(self, division: str, model_info: dict, param_grid: dict, date: dict):
        # Data Configuration
        self.division = division    # SELL-IN / SELL-OUT
        self.date = date
        self.col_target = 'qty'     # Target column
        self.col_exo = ['discount']     # Exogenous features
        self.hrchy_level = config.HRCHY_LEVEL  # Data Hierarchy

        # Algorithm Configuration
        self.algorithm = Algorithm()
        self.cand_models = list(model_info.keys())
        self.param_grid = param_grid
        self.model_to_variate = config.MODEL_TO_VARIATE
        self.model_info = model_info
        self.model_fn = {'ar': self.algorithm.ar,
                         'arima': self.algorithm.arima,
                         'hw': self.algorithm.hw,
                         'var': self.algorithm.var,
                         'varmax': self.algorithm.varmax,
                         'sarima': self.algorithm.sarimax}

        # Training Configuration
        self.validation_method = config.VALIDATION_METHOD
        self.n_test = config.N_TEST

    def train(self, df) -> dict:
        scores = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                      fn=self.train_model,
                                      df=df)

        return scores

    def train_model(self, df) -> List[List[np.array]]:
        feature_by_variable = self.select_feature_by_variable(df=df)

        models = []
        for model in self.cand_models:
            score = self.validation(data=feature_by_variable[self.model_to_variate[model]], model=model)
            models.append([model, np.round(score, 3)])
        models = sorted(models, key=lambda x: x[1])

        return models

    def select_feature_by_variable(self, df: pd.DataFrame):
        feature_by_variable = {'univ': df[self.col_target],
                               'multi': df[self.col_exo + [self.col_target]]}

        return feature_by_variable

    def validation(self, data, model: str):
        score = 0
        # if len(data) > int(self.model_info[model]['input_width']):
        if self.validation_method == 'train_test':
            score = self.train_test_validation(data=data, model=model)

        elif self.validation_method == 'walk_forward':
            score = self.walk_fwd_validation(data=data, model=model)

        return score

    def make_score_result(self, scores: dict) -> pd.DataFrame:
        result = util.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_level,
                                               fn=self.score_to_df,
                                               df=scores)

        result = pd.DataFrame(result)
        cols = ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'rmse']
        result.columns = cols

        result['project_cd'] = 'ENT001'
        result['data_vrsn_cd'] = self.date['date_from'] + '-' + self.date['date_to']
        result['division'] = self.division
        result['fkey'] = ['HRCHY' + str(i+1) for i in range(len(result))]
        result['rmse'] = result['rmse'].fillna(0)

        return result

    @staticmethod
    def score_to_df(hrchy: list, data) -> List[list]:
        result = []
        for algorithm, score in data:
            result.append(hrchy + [algorithm, score])

        return result

    def train_test_validation(self, model: str, data) -> np.array:
        # split dataset
        data_train = None
        data_test = None
        data_length = len(data)
        if self.model_to_variate[model] == 'univ':
            data_train = data.iloc[: data_length - self.n_test]
            data_test = data.iloc[data_length - self.n_test:]
        elif self.model_to_variate[model] == 'multi':
            data_train = data.iloc[: data_length - self.n_test, :]
            data_train = {'endog': data_train[self.col_target].values.ravel(),
                          'exog': data_train[self.col_exo].values.ravel()}
            data_test = data.iloc[data_length - self.n_test:, :]
            data_test = {'endog': data_test[self.col_target],
                         'exog': data_test[self.col_exo].values.ravel()}

        # evaluation
        yhat = self.model_fn[model](history=data_train, cfg=self.param_grid[model], pred_step=self.n_test)
        yhat = np.nan_to_num(yhat)

        err = 0
        if self.model_to_variate[model] == 'univ':
            err = mean_squared_error(data_test, yhat, squared=False)
        elif self.model_to_variate[model] == 'multi':
            err = mean_squared_error(data_test['endog'], yhat, squared=False)

        return err

    def walk_fwd_validation(self, model: str, data) -> np.array:
        """
        :param model: Statistical model
        :param data: time series data
        :param params: configuration
        :param n_test: number of test data
        :return:
        """
        # split dataset
        dataset = self.window_generator(df=data, model=model)

        # evaluation
        predictions = []
        for train, test in dataset:
            yhat = self.model_fn[model](history=train, cfg=self.param_grid[model], pred_step=self.n_test)
            yhat = np.nan_to_num(yhat)
            err = mean_squared_error(test, yhat, squared=False)
            predictions.append(err)

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def window_generator(self, df, model: str) -> List:
        data_length = len(df)
        input_width = int(self.model_info[model]['input_width'])
        label_width = int(self.model_info[model]['label_width'])
        data_input = None
        data_target = None
        dataset = []
        for i in range(data_length - input_width - label_width + 1):
            if self.model_to_variate[model] == 'univ':
                data_input = df.iloc[i: i + input_width]
                data_target = df.iloc[i + input_width: i + input_width + label_width]
            elif self.model_to_variate[model] == 'multi':
                data_input = df.iloc[i: i + input_width, :]
                data_target = df.iloc[i + input_width: i + input_width + label_width, :]

            dataset.append((data_input, data_target))

        return dataset

    def train_test_split(self, data, model):
        train_rate = config.TRAIN_RATE
        data_length = len(data)
        if self.model_to_variate[model] == 'univ':
            return data[: int(data_length * train_rate)], data[int(data_length * train_rate):]

        elif self.model_to_variate[model] == 'multi':
            return data.iloc[:int(data_length * train_rate), :], data.iloc[int(data_length * train_rate):, :]

    def score_model(self, model: str, data, n_test, cfg) -> tuple:
        # convert config to a key
        key = str(cfg)
        result = self.walk_fwd_validation(model=model, data=data, n_test=n_test, params=cfg)

        return model, key, result

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
