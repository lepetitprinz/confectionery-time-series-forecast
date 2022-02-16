import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccuracy import PipelineAccuracy

hist_to = '20220130'        # W05(20220130) / W04(20220123)
division_list = ['SELL_IN', 'SELL_OUT']
item_lvl_list = [3, 5]
load_option = 'db'

exec_cfg = {
    'cls_prep': True,     # Preprocessing
    'cls_comp': True,     # Compare result
    'cls_top_n': False,    # Choose top N
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

pipe_acc = PipelineAccuracy(
    hist_to=hist_to,
    division_list=division_list,
    item_lvl_list=item_lvl_list
)
