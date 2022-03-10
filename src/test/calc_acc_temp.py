import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccuracyNew import PipelineAccuracy

hist_to = '20220220'    # W09(20220220) / W08(20220213) / W07(20220206) / W06(20220130)
exec_kind = 'dev'
item_lvl_list = [5]
division_list = ['SELL_OUT']    # SELL_IN / SELL_OUT

acc_classify_standard = 0.25

root_path = os.path.join('..', '..')
save_path = os.path.join(root_path, 'analysis', 'accuracy', exec_kind)


exec_cfg = {
    'save_db_yn': True,
    'cycle_yn': False,
    'rm_zero_yn': False,                   # Remove zeros
    'filter_sales_threshold_yn': False,    # Filter based on sales threshold
    'pick_specific_biz_yn': False,         # Pick Specific business code
    'pick_specific_sp1_yn': False,        # Pick Specific sp1 list
}

pipe_acc = PipelineAccuracy(
    exec_kind=exec_kind,
    exec_cfg=exec_cfg,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
    hist_to=hist_to,
    acc_classify_standard=acc_classify_standard
)
print(f"Apply end date of history: {hist_to}")
pipe_acc.run()
print("Calculating accuracy is finished")
