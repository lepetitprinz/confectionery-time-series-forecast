import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

# hist_from = '20190204'    # W05(20190204) / W04(20190128)
hist_to = '20220130'        # W05(20220130) / W04(20220123)

# Change data type (string -> datetime)
hist_to_datetime = datetime.datetime.strptime(hist_to, '%Y%m%d')

# Add dates
hist_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=156) + datetime.timedelta(days=1)
# compare_from = hist_to_datetime + datetime.timedelta(days=1)
# compare_to = hist_to_datetime + datetime.timedelta(days=7)
compare_from = hist_to_datetime + datetime.timedelta(days=8)
compare_to = hist_to_datetime + datetime.timedelta(days=14)

# Change data type (datetime -> string)
hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
compare_from = datetime.datetime.strftime(compare_from, '%Y%m%d')
compare_to = datetime.datetime.strftime(compare_to, '%Y%m%d')

exec_cfg = {
    'cls_prep': True,     # Preprocessing
    'cls_comp': True,     # Compare result
    'cls_top_n': True,    # Choose top N
    'cls_graph': False    # Draw graph
}

opt_cfg = {
    'rm_zero_yn': True,                   # Remove zeros
    'calc_acc_by_sp1_item_yn': True,      # Calculate accuracy on SP1 items
    'filter_sales_threshold_yn': True,    # Filter based on sales threshold
    'filter_specific_acc_yn': False,      # Filter Specific accuracy range
    'pick_specific_biz_yn': True,         # Pick Specific business code
    'pick_specific_sp1_yn': False,        # Pick Specific sp1 list
}

data_cfg = {
    # 'root_path': os.path.join('..', '..'),
    'root_path':  os.path.join('/', 'opt', 'DF', 'fcst'),
    'item_lvl': 3,
    'division': 'SELL_IN',  # SELL_IN / SELL_OUT
    'load_option': 'db',  # db / csv
    'item_attr01_cd': 'P1'
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
