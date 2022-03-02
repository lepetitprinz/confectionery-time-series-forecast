import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

hist_to = '20220213'    # W08(20220213) / W07(20220206) / W06(20220130)
exec_kind = 'dev'
root_path = os.path.join('..', '..')
# root_path = os.path.join('/', 'opt', 'DF', 'fcst')
save_path = os.path.join(root_path, 'analysis', 'accuracy', 'batch')

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

step_cfg = {
    'cls_prep': True,     # Preprocessing
    'cls_comp': True,     # Compare result
    'cls_top_n': False,    # Choose top N
    'cls_graph': False    # Draw graph
}

exec_cfg = {
    'rm_zero_yn': True,                   # Remove zeros
    'calc_acc_by_sp1_item_yn': False,     # Calculate accuracy on SP1 items
    'filter_sales_threshold_yn': True,    # Filter based on sales threshold
    'filter_specific_acc_yn': False,      # Filter Specific accuracy range
    'pick_specific_biz_yn': False,         # Pick Specific business code
    'pick_specific_sp1_yn': False,        # Pick Specific sp1 list
}

data_cfg = {
    'root_path': root_path,
    'save_path': save_path,
    'item_lvl': 3,
    'division': 'SELL_OUT',  # SELL_IN / SELL_OUT
    'load_option': 'csv',  # db / csv
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
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    date_cfg=date_cfg,
    data_cfg=data_cfg,
    exec_kind=exec_kind
)

# run
comp.run()
