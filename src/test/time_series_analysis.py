from baseline.analysis.TimeSeriesAnalysis import TimeSeriesAnalysis

import os

exec_cfg = {
}

data_cfg = {
    'root_path': os.path.join('..', '..', 'analysis', 'accuracy'),
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'data_version': '20190114-20220109',
    'target_col': 'sales'
}

tsa = TimeSeriesAnalysis(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg
)

tsa.run()
