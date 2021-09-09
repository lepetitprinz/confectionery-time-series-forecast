import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


class Profiling(object):
    def __init__(self, item_col: str, meta_col: str,
                 save_step_yn=False, load_step_yn=False, save_db_yn=False):
        self.item_col = item_col
        self.meta_col = meta_col
        self.item_indices = {}
        self.vectorize_method = {'count_vectorizer': CountVectorizer(),
                                 'tfidf_vectorizer': TfidfVectorizer()}    # count_vectorizer / tfidf_vectorizer

        # Save & Load Configuration
        self.save_steps_yn = save_step_yn
        self.load_step_yn = load_step_yn
        self.save_db_yn = save_db_yn

    def profiling(self, data: pd.DataFrame):
        meta_data = self.make_meta_data(data=data)
        matrix = self.vectorize(meta_data=meta_data, method='tfidf_vectorizer')
        similarity = self.calc_similarity(matrix=matrix)

        return similarity

    def make_meta_data(self, data: pd.DataFrame):
        meta_data = data.groupby(by=[self.item_col]).agg({self.meta_col: lambda x: ' '.join(x)})
        meta_data = meta_data.reset_index(level=0)
        item_indices = self.make_item_indices(items=list(meta_data[self.item_col]))
        self.item_indices = item_indices

        return meta_data

    def vectorize(self, meta_data: np.array, method: str):
        vectorizer = self.vectorize_method[method]
        matrix = vectorizer.fit_transform(meta_data[self.meta_col])

        return matrix

    @staticmethod
    def calc_similarity(matrix):
        similarity = cosine_similarity(matrix, matrix)

        return similarity

    def make_item_indices(self, items):
        item_indices = {item: i for i, item in enumerate(items)}

        return item_indices
