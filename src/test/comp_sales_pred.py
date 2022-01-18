import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.PredCompare import PredCompare

hist_from = '20190114'
hist_to = '20220109'

data_cfg = {
    'root_path': os.path.join('..', '..', 'analysis', 'accuracy'),
    'division': 'SELL_OUT',
    'item_lvl': 3,
    'rm_zero_yn': True,
    'filter_sales_threshold_yn': True,
    'pick_specific_sp1_yn': True,
    'draw_plot_yn': True
}

date_cfg = {
    'cycle_yn': False,
    'date': {
        'hist': {
            'from': hist_from,
            'to': hist_to
        },
        'compare': {
            'from': '20220110',
            'to': '20220116'
        }
    },
    'data_vrsn_cd': hist_from + '-' + hist_to
}

# Initialize class
comp = PredCompare(date_cfg=date_cfg, data_cfg=data_cfg)

# run
comp.run()
