import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.TimeSeriesAnalysis import TimeSeriesAnalysis

exec_cfg = {
}

data_cfg = {
    'root_path': os.path.join('..', '..', 'analysis'),
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'data_version': '20190121-20220116',
    'target_col': 'sales'
}

tsa = TimeSeriesAnalysis(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg
)

tsa.run()