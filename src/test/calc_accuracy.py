import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

hist_from = '20190121'
hist_to = '20220116'

compare_from = '20220117'
compare_to = '20220123'

exec_cfg = {
    'cls_prep': True,    # Preprocessing
    'cls_comp': True,    # Compare result
    'cls_top_n': True,   # Choose top N
    'cls_graph': True   # Draw graph
}

opt_cfg = {
    'rm_zero_yn': True,    # Preprocessing
    'calc_acc_by_sp1_item_yn': True,
    'filter_sales_threshold_yn': True,    # Preprocessing
    'filter_specific_acc_yn': False,    # Compare result
    'filter_sepcific_biz_yn': True,
    'pick_specific_sp1_yn': True,    # Top N
}

data_cfg = {
    'root_path': os.path.join('..', '..', 'analysis', 'accuracy'),
    'division': 'SELL_OUT',
    'item_lvl': 3,
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
            'to': compare_to    # 20220116
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
