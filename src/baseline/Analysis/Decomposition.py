from dao.DataIO import DataIO

import pandas as pd
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


class Decomposition(object):
    def __init__(self, division: str, hrchy_lvl_cd: str):
        self.division = division
        self.hrchy_lvl_cd = hrchy_lvl_cd
        self.x = 'qty'
        self.model = 'additive'     # additive / multiplicative
        self.tb_name = 'M4S_O110500'
        self.dao = DataIO()

    def decompose(self, df):
        date = pd.to_datetime(df['yymmdd'], format='%Y%m%d')
        data = pd.Series(data=df[self.x].to_numpy(), index=date.to_numpy())
        data_resampled = data.resample(rule='D').sum()
        decomposed = seasonal_decompose(x=data_resampled, model=self.model)
        item_info = df[['biz_cd', 'line_cd', 'brand_cd', 'item_ctgr_cd']].drop_duplicates()
        item_info = item_info.iloc[0].to_dict()
        result = pd.DataFrame(
            {'project_cd': 'ENT001',
             'division_cd': self.division,
             'hrchy_lvl_cd': self.hrchy_lvl_cd,
             'biz_cd': item_info.get('biz_cd', ''),
             'line_cd': item_info.get('line_cd', ''),
             'brand_cd': item_info.get('brand_cd', ''),
             'item_cd': item_info.get('item_ctgr_cd', ''),
             'yymmdd': [datetime.strftime(dt, '%Y%m%d') for dt in list(data_resampled.index)],
             'org_val': decomposed.observed,
             'trend_val': decomposed.trend.fillna(0),
             'seasonal_val': decomposed.seasonal.fillna(0),
             'resid_val': decomposed.resid.fillna(0),
             'create_user_cd': 'SYSTEM',
             'create_date': datetime.now()})

        # self.dao.update_to_db(df=result, tb_name=self.tb_name)
