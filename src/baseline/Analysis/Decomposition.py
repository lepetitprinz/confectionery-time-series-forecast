import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


class Decomposition(object):
    def __init__(self, common, division: str, hrchy: dict, date_range):
        self.date_range = date_range
        self.division = division
        self.hrchy = hrchy
        self.x = common['target_col']
        self.model = 'additive'     # additive / multiplicative
        self.tb_name = 'M4S_O110500'
        self.save_to_db_yn = False

    def decompose(self, df):
        data = pd.Series(data=df[self.x].to_numpy(), index=df.index)

        data_resampled = data.resample(rule='W').sum()
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
                'project_cd': 'ENT001',
                'division_cd': self.division,
                'hrchy_lvl_cd': self.hrchy['key'][:-1],
                'item_attr01_cd': item_info.get('biz_cd', ''),
                'item_attr02_cd': item_info.get('line_cd', ''),
                'item_attr03_cd': item_info.get('brand_cd', ''),
                'item_attr04_cd': item_info.get('item_cd', ''),
                'yymmdd': [datetime.strftime(dt, '%Y%m%d') for dt in list(data_resampled.index)],
                'org_val': decomposed.observed,
                'trend_val': np.round(decomposed.trend.fillna(0), 1),
                'seasonal_val': np.round(decomposed.seasonal.fillna(0), 1),
                'resid_val': np.round(decomposed.resid.fillna(0), 1),
                'create_user_cd': 'SYSTEM'
            })
            return result

        except ValueError:
            return None
