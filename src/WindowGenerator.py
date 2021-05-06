import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras


class WindowGenerator(object):
    """
    Main features
    - width(number of time steps)
    - time offset: prediction steps
    - Which features are used as inputs, labels, or both
    Dataset for deep learning model(DNN, CNN, RNN, ...)
    - Single-output, and multi-output predictions
    - Single-time-step and multi-time-step predictions
    """
    def __init__(self, input_width: int, label_width: int, shift: int,
                 train_df: pd.DataFrame, val_df: pd.DataFrame, label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width      # history interval used in training
        self.label_width = label_width      # prediction interval
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        Convert data to a window of inputs and a window of labels
        :param features:
        :return:
        """
        inputs = features.to_numpy()[:, self.input_slice]
        labels = features.to_numpy()[:, self.labels_slice]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels