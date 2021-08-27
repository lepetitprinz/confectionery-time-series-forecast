import common.config as config

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


class Decomposition(object):
    def __init__(self):
        self.model = 'additive'     # additive / multiplicative
        self.x = 'qty'

        # Hierarchy
        self.hrchy_list = config.HRCHY_LIST
        self.hrchy = config.HRCHY
        self.hrchy_level = config.HRCHY_LEVEL

    def decompose(self, df=None, val=None, lvl=0):
        temp = None
        if lvl == 0:
            temp = {}
            for key, val in df.items():
                result = self.decompose(val=val, lvl=lvl + 1)
                temp[key] = result

        elif lvl < self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                result = self.decompose(val=val_hrchy, lvl=lvl + 1)
                temp[key_hrchy] = result

            return temp

        elif lvl == self.hrchy_level:
            temp = {}
            for key_hrchy, val_hrchy in val.items():
                if len(val_hrchy) > 2:
                    date = pd.to_datetime(val_hrchy['yymmdd'], format='%Y%m%d')
                    data = pd.Series(data=val_hrchy[self.x].to_numpy(), index=date.to_numpy())
                    data_resampled = data.resample(rule='D').sum()
                    decomposed = seasonal_decompose(x=data_resampled, model=self.model)
                    result = pd.DataFrame(
                        {'date': data_resampled.index,
                         'observed': decomposed['original'],
                         'trend': decomposed['trend'],
                         'seasonal': decomposed['seasonal'],
                         'residual': decomposed['residual']})
                    temp[key_hrchy] = result

            return temp

        return temp