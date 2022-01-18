import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.SalesAnalysis import SalesAnalysis

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_prep': True,
    'cls_comp': True,
    'cls_view': False
}

data_cfg = {
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'cycle_yn': False,
    'rm_zero_yn': True,
    'date': {
        'from': '20190107',
        'to': '20220102'
    }
}

sa = SalesAnalysis(step_cfg=step_cfg, data_cfg=data_cfg)
sa.run()
