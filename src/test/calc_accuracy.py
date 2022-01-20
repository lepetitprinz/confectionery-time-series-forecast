import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

hist_from = '20190114'
hist_to = '20220109'

exec_cfg = {
    'cls_prep': True,    # Preprocessing
    'cls_comp': True,    # Compare result
    'cls_top_n': False,   # Choose top N
    'cls_graph': False   # Draw graph
}

opt_cfg = {
    'rm_zero_yn': True,    # Preprocessing
    'filter_sales_threshold_yn': True,    # Preprocessing
    'filter_specific_acc_yn': False,    # Compare result
    'pick_specific_sp1_yn': True,    # Top N
}

data_cfg = {
    'root_path': os.path.join('..', '..', 'analysis', 'accuracy'),
    'division': 'SELL_OUT',
    'item_lvl': 3,
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
comp = CalcAccuracy(exec_cfg=exec_cfg, opt_cfg=opt_cfg, date_cfg=date_cfg, data_cfg=data_cfg)

# run
comp.run()
