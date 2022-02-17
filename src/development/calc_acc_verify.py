import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccuracy import PipelineAccuracy

exec_kind = 'verify'
pred_load_option = 'csv'
item_lvl_list = [3, 5]
division_list = ['SELL_IN']    # ['SELL_OUT']
root_path = os.path.join('/', 'opt', 'DF', 'fcst')
save_path = os.path.join(root_path, 'analysis', 'accuracy', 'verify')

step_cfg = {
    'cls_prep': True,     # Preprocessing
    'cls_comp': True,     # Compare result
    'cls_top_n': False,    # Choose top N
    'cls_graph': False    # Draw graph
}

exec_cfg = {
    'cycle_yn': True,
    'rm_zero_yn': True,                   # Remove zeros
    'filter_sales_threshold_yn': True,    # Filter based on sales threshold
    'pick_specific_biz_yn': False,         # Pick Specific business code
    'pick_specific_sp1_yn': False,        # Pick Specific sp1 list
}

pipe_acc = PipelineAccuracy(
    exec_kind=exec_kind,
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    pred_load_option=pred_load_option,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
)

pipe_acc.run()