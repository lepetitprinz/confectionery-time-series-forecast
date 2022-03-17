import numpy as np


class WindowGenerator(object):
    def __init__(self, input_width, target_width, shift_width):
        self.input_width = input_width
        self.target_width = target_width
        self.shift_width = shift_width

        self.total_window_size = input_width + shift_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arnage(self.total_window_size)[self.input_slice]

        self.target_start = self.total_window_size - self.target_width
        self.target_slice = slice(self.target_start, None)
        self.target_indices = np.arange(self.total_window_size)[self.target_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.target_indices}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        target = features[:, self.target_slice, :]
