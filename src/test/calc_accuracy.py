import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

hist_from = '20190128'
hist_to = '20220123'

compare_from = '20220124'
compare_to = '20220130'

exec_cfg = {
    'cls_prep': True,    # Preprocessing
    'cls_comp': True,    # Compare result
    'cls_top_n': True,   # Choose top N
    'cls_graph': False    # Draw graph
}

opt_cfg = {
    'rm_zero_yn': True,    # Preprocessing
    'calc_acc_by_sp1_item_yn': True,
    'filter_sales_threshold_yn': True,    # Filter based on sales threshold
    'filter_specific_acc_yn': False,      # Filter Specific accuracy range
    'pick_specific_biz_yn': True,         # Pick Specific business code
    'pick_specific_sp1_yn': False,        # Pick Specific sp1 list
}

data_cfg = {
    # 'root_path': os.path.join('..', '..'),
    'root_path':  os.path.join('/', 'opt', 'DF', 'fcst'),
    'load_option': 'csv',
    'division': 'SELL_IN',
    'item_lvl': 5,
    'item_attr01_cd': 'P2'
}

date_cfg = {
    'cycle_yn': False,
    'date': {
        'hist': {
            'from': hist_from,
            'to': hist_to
        },
        'compare': {
            'from': compare_from,    # 20220110
            'to': compare_to         # 20220116
        }
    },
    'data_vrsn_cd': hist_from + '-' + hist_to
}

# Initialize class
comp = CalcAccuracy(
    exec_cfg=exec_cfg,
    opt_cfg=opt_cfg,
    date_cfg=date_cfg,
    data_cfg=data_cfg
)

# run
comp.run()
