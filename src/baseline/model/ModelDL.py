from baseline.preprocess.WindowGenerator import WindowGenerator

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Lambda, Conv1D, Reshape
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class Models(object):
    def __init__(self, window: WindowGenerator, cfg: dict, common):
        self.window = window
        self.n_features = window.train_df.shape[-1]
        self.window_input = window.input_width
        self.window_output = window.label_width

        # Model Hyper-parameters
        self.n_hidden = cfg['n_hidden']   # Seq to Seq Model Units
        self.activation = 'relu'    # Activation Function
        self.dropout = 0.1    # Dropout rate
        self.recurrent_dropout = 0.1
        self.momentum = 0.99    # Batch Normalization hyper-parameters
        self.lr = 0.001    # Learning Rate

        # convolutional model
        self.conv_width = 3

        # Training hyper-parameters
        self.validation_split = 0.2
        self.batch_size = 64
        self.epochs = cfg['epochs']

        self.target_col = common['target_col']
        self.save_path = ''

    #########################
    # Time Series LSTM Model
    #########################
    def vanilla(self) -> Model:
        # define vanilla LSTM model
        # n time-steps to one time-step
        # Many-to-one model
        model = Sequential()
        model.add(LSTM(32, activation=self.activation, input_shape=(self.window_input, self.n_features),
                       return_sequences=True))
        model.add(LSTM(16, activation=self.activation, return_sequences=True))
        model.add(LSTM(8, activation=self.activation, return_sequences=False))
        model.add(Dense(1))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def stacked(self) -> Model:
        # define stacked LSTM  model
        # Multi variate to ta rget
        # n time-steps to n time-steps
        model = Sequential()
        model.add(LSTM(32, activation=self.activation, input_shape=(self.window_input, self.n_features),
                       return_sequences=True))
        model.add(LSTM(16, activation=self.activation, return_sequences=True))
        model.add(LSTM(8, activation=self.activation, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def encoder_decoder(self) -> Model:
        # define encoder-decoder LSTM model
        # Many-to-one model
        model = Sequential()
        model.add(LSTM(64, activation=self.activation, return_sequences=True,
                       input_shape=(self.window_input, self.n_features)))
        model.add(LSTM(32, activation=self.activation, return_sequences=False))
        model.add(RepeatVector(self.window_input))
        model.add(LSTM(32, activation=self.activation, return_sequences=True))
        model.add(LSTM(64, activation=self.activation, return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_features)))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def seq2seq(self) -> Model:
        """
        Sequence to sequence model
        # Multivariate to target
        # m time-steps to n time-steps
        :return:
        """
        # define sequence to sequence LSTM model
        input_train = Input(shape=(self.window_input, self.n_features))
        output_train = Input(shape=(self.window_output, self.n_features))

        # Encoder LSTM
        encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
            self.n_hidden, activation=self.activation,
            dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
            return_sequences=False, return_state=True)(input_train)

        # Batch normalization
        encoder_last_h1 = BatchNormalization(momentum=self.momentum)(encoder_last_h1)
        encoder_last_c = BatchNormalization(momentum=self.momentum)(encoder_last_c)

        # Decoder LSTM
        decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
        decoder = LSTM(self.n_hidden, activation=self.activation, dropout=self.dropout,
                       recurrent_dropout=self.recurrent_dropout, return_state=False,
                       return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
        out = TimeDistributed(Dense(1))(decoder)

        model = Model(inputs=input_train, outputs=out)

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def multi_linear(self):
        model = Sequential()
        model.add(Lambda(lambda x: x[:, -1:, :]))
        model.add(Dense(self.window_output * self.n_features, kernel_initializer=tf.initializers.zeros()))
        model.add(Reshape([self.window_output, self.n_features]))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def multi_dense(self):
        model = Sequential()
        model.add(Lambda(lambda x: x[:, -1:, :]))
        model.add(Dense(128, activation=self.activation))
        model.add(Dense(self.window_output * self.n_features, kernel_initializer=tf.initializers.zeros()))
        model.add(Reshape([self.window_output, self.n_features]))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def multi_conv(self):
        """
        Convolutional Neural Network model
        :return:
        """
        model = Sequential()
        model.add(Lambda(lambda x: x[:, -self.conv_width:, :]))
        model.add(Conv1D(128, activation=self.activation, kernel_size=(self.conv_width)))
        model.add(Dense(self.window_output * self.n_features, kernel_initializer=tf.initializers.zeros()))
        model.add(Reshape([self.window_output, self.n_features]))

        optimizer = Adam(lr=self.lr, clipnorm=1)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        model.summary()

        return model

    def train(self, model: Model, save_nm: str):
        # define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)

        # train
        history = model.fit(self.window.train, epochs=self.epochs,
                            validation_data=self.window.val,
                            verbose=1, callbacks=[early_stopping])

        # save model
        model.save(os.path.join(self.save_path, save_nm + "h5"))

        return history

    def evaluation(self, model: Model):
        eval_val = model.evaluate(self.window.val)
        eval_test = model.evaluate(self.window.test)

        return eval_val, eval_test

    def predict(self, model: Model, data: pd.DataFrame, normalize_stats: tuple):
        mean, std = normalize_stats
        data_sliced = data.iloc[-self.window.input_width:, :]
        data_norm = (data_sliced - mean) / std
        data_reshape = data_norm.to_numpy().reshape((1, data_sliced.shape[0], data_sliced.shape[1]))

        yhat = model.predict(data_reshape)
        yhat = yhat * std[self.target_col] + mean[self.target_col]

        return yhat