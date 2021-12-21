import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


class Decomposition(object):
    def __init__(self, common, hrchy: dict, date: dict):
        self.resample_rule = 'D'
        self.date_range = pd.date_range(
            start=date['history']['from'],
            end=date['history']['to'],
            freq=self.resample_rule
        )
        self.hrchy = hrchy
        self.x = common['target_col']
        self.model = 'additive'     # additive / multiplicative
        self.tb_name = 'M4S_O110500'

    def decompose(self, df):
        data = pd.Series(data=df[self.x].to_numpy(), index=df.index)

        data_resampled = data.resample(rule=self.resample_rule).sum()
        # data_resampled = data.resample(rule='D').sum()

        if len(data_resampled.index) != len(self.date_range):
            idx_add = list(set(self.date_range) - set(data_resampled.index))
            data_add = np.zeros(len(idx_add))
            df_add = pd.Series(data_add, index=idx_add)
            data_resampled = data_resampled.append(df_add)
            data_resampled = data_resampled.sort_index()

        data_resampled = data_resampled.fillna(0)

        # Seasonal Decomposition
        try:
            decomposed = seasonal_decompose(x=data_resampled, model=self.model)
            item_info = df[self.hrchy['apply']].drop_duplicates()
            item_info = item_info.iloc[0].to_dict()
            result = pd.DataFrame({
                'item_attr01_cd': item_info.get('biz_cd', ''),
                'item_attr02_cd': item_info.get('line_cd', ''),
                'item_attr03_cd': item_info.get('brand_cd', ''),
                'item_attr04_cd': item_info.get('item_cd', ''),
                'yymmdd': [datetime.strftime(dt, '%Y%m%d') for dt in list(data_resampled.index)],
                'org_val': decomposed.observed,
                'trend_val': np.round(decomposed.trend.fillna(0), 1),
                'seasonal_val': np.round(decomposed.seasonal.fillna(0), 1),
                'resid_val': np.round(decomposed.resid.fillna(0), 1),
            })
            return result

        except ValueError:
            return None
