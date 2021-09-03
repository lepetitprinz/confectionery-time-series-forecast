import common.util as util
import common.config as config
from baseline.model.Algorithm import Algorithm

from datetime import datetime
from datetime import timedelta
import pandas as pd


class Predict(object):
    def __init__(self, division: str, model_info: dict, param_grid: dict, end_date):
        self.division = division    # SELL-IN / SELL-OUT
        self.col_target = 'qty'

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

    def forecast(self, df=None, val=None, lvl=0, hrchy=[]):
        if lvl == 0:
            temp = []
            for key, val in df.items():
                hrchy.append(key)
                result = self.forecast(val=val, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key)

        elif lvl < self.hrchy_level:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = self.forecast(val=val_hrchy, lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        elif lvl == self.hrchy_level:
            for key_hrchy, val_hrchy in val.items():
                if len(val_hrchy) > self.n_test:
                    hrchy.append(key_hrchy)
                    temp = []
                    for algorithm in self.cand_models:
                        prediction = self.model_fn[algorithm](history=val_hrchy[self.col_target].to_numpy(),
                                                              cfg=self.param_grid[algorithm],
                                                              pred_step=self.n_test)
                        temp.append(hrchy + [algorithm, prediction])
                    hrchy.remove(key_hrchy)

            return temp

        return temp

    def make_pred_result(self, df):
        end_date = datetime.strptime(self.end_date, '%Y%m%d')

        results = []
        fkey = ['HRCHY' + str(i + 1) for i in range(len(df))]
        for i, pred in enumerate(df):
            for j, result in enumerate(pred[-1]):
                results.append([fkey[i]] + pred[:-1] +
                               [datetime.strftime(end_date + timedelta(weeks=(j + 1)), '%Y%m%d'), result])

        results = pd.DataFrame(results)
        cols = ['fkey'] + ['S_COL0' + str(i + 1) for i in range(self.hrchy_level + 1)] + ['stat', 'month',
                                                                                          'result_sales']
        results.columns = cols
        results['project_cd'] = 'ENT001'
        results['division'] = self.division

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