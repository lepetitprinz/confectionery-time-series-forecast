import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class FeatureEngineering(object):
    def __init__(self, common, exg_list):
        self.common = common
        self.target_col = common['target_col']
        self.exg_list = exg_list
        self.feat_select_method = 'spearmanr'    # pearson / spearmanr
        self.n_feature_to_select = 2

    def feature_selection(self, data: pd.DataFrame):
        drop_feat_list, exg_list = self.numeric_to_numeric(data=data)
        data = data.drop(columns=drop_feat_list)

        return data, exg_list

    def numeric_to_numeric(self, data):
        # feature selection with numeric to numeric
        target = data[self.target_col].values

        coef_list = []
        for exg in self.exg_list:
            coef = 0
            if self.feat_select_method == 'spearmanr':
                coef, p = spearmanr(target, data[exg].values)
            elif self.feat_select_method == 'pearson':
                coef = np.corrcoef(target, data[exg].values)[0][1]
            coef_list.append((exg, abs(coef)))

         # Rank the feature importance
        coef_list = sorted(coef_list, key=lambda x: x[1], reverse=True)
        exg_list = coef_list[:self.n_feature_to_select]    # Feature select
        drop_exg_list = coef_list[self.n_feature_to_select:]

        exg_list = [exg for exg, p in exg_list]
        drop_exg_list = [exg for exg, p in drop_exg_list]

        return drop_exg_list, exg_list
