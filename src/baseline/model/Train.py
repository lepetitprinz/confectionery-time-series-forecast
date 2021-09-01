import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

import warnings
import numpy as np
import pandas as pd
from math import sqrt
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import mean_squared_error

# Tensorflow library
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')


class Train(object):
    def __init__(self, division: str, model_info: dict, param_grid: dict, end_date):
        # Data Configuration
        self.division = division    # SELL-IN / SELL-OUT
        self.end_date = end_date
        self.col_target = 'qty'
        self.col_exo = ['discount', '']

        # Hierarchy
        self.hrchy_list = config.HRCHY_LIST
        self.hrchy = config.HRCHY
        self.hrchy_level = config.HRCHY_LEVEL

        self.n_test = config.N_TEST

        # Algorithms
        self.algorithm = Algorithm()
        self.cand_models = list(model_info.keys())
        self.param_grid = param_grid
        self.model_to_variate = config.MODEL_TO_VARIATE
        self.model_fn = {'ar': self.algorithm.ar,
                         'arima': self.algorithm.arima,
                         'hw': self.algorithm.hw,
                         'var': self.algorithm.var,
                         'varmax': self.algorithm.varmax}

        self.model_width = model_info

        self.epoch_best = 0

    def train(self, df) -> dict:
        scores = util.hrchy_recursion(hrchy_lvl=self.hrchy_level,
                                      fn=self.train_model,
                                      df=df)

        return scores

    def make_score_result(self, scores: dict) -> pd.DataFrame:
        result = util.hrchy_recursion_with_key(hrchy_lvl=self.hrchy_level,
                                               fn=self.score_to_df,
                                               df=scores)

        result = pd.DataFrame(result)
        cols = ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'rmse']
        result.columns = cols

        result['project_cd'] = 'ENT001'
        result['division'] = self.division
        result['fkey'] = ['HRCHY' + str(i+1) for i in range(len(result))]

        result['rmse'] = result['rmse'].fillna(0)

        return result

    @staticmethod
    def score_to_df(hrchy: list, data):
        result = []
        for algorithm, score in data:
            result.append(hrchy + [algorithm, score])

        return result

    def make_pred_result(self, predictions) -> pd.DataFrame:
        end_date = datetime.strptime(self.end_date, '%Y%m%d')

        results = []
        fkey = ['HRCHY' + str(i+1) for i in range(len(predictions))]
        for i, pred in enumerate(predictions):
            for j,  result in enumerate(pred[-1]):
                results.append([fkey[i]] + pred[:-1] +
                              [datetime.strftime(end_date + timedelta(weeks=(j+1)), '%Y%m%d'), result])

        results = pd.DataFrame(results)
        cols = ['fkey'] + ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'month', 'result_sales']
        results.columns = cols
        results['project_cd'] = 'ENT001'
        results['division'] = self.division

        return results

    def train_model(self, df) -> list:
        models = []
        for model in self.cand_models:
            data = self.filter_data(df=df, model=model)
            score = self.walk_fwd_validation(model=model, cfg=self.param_grid[model],
                                             data=data, n_test=self.n_test)
            models.append([model, np.round(score, 2)])

        models = sorted(models, key=lambda x: x[1])

        return models

    def filter_data(self, df: pd.DataFrame, model: str):
        df = df.set_index(keys='yymmdd')    # Todo : Check index
        filtered = None
        if self.model_to_variate[model] == 'univ':
            filtered = df[self.col_target]
        elif self.model_to_variate[model] == 'multi':
            filtered = df[self.col_exo + [self.col_target]]

        return filtered

    def walk_fwd_validation(self, model: str, data, cfg, n_test) -> np.array:
        """
        :param model: Statistical model
        :param data: time series data
        :param cfg: configuration
        :param n_test: number of test data
        :return:
        """
        # split dataset
        dataset = self.make_ts_dataset(df=data, model=model)

        predictions = []
        for train, test in dataset:
            yhat = self.model_fn[model](history=train, cfg=cfg, pred_step=n_test)
            err = self.calc_sqrt_mse(test, yhat)
            predictions.append(err)

        # estimate prediction error
        rmse = np.mean(predictions)

        return rmse

    def make_ts_dataset(self, df, model: str):
        data_length = len(df)
        input_width = int(self.model_width[model]['input_width'])
        label_width = int(self.model_width[model]['label_width'])
        data_input = None
        data_target = None
        dataset = []
        for i in range(data_length - label_width + 1):
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

    @staticmethod
    def calc_sqrt_mse(actual, predicted) -> float:
        return sqrt(mean_squared_error(actual, predicted))

    def score_model(self, model: str, data, n_test, cfg) -> tuple:
        # convert config to a key
        key = str(cfg)
        result = self.walk_fwd_validation(model=model, data=data, n_test=n_test, cfg=cfg)

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
