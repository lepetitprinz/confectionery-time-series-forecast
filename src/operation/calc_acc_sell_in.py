import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccuracy import PipelineAccuracy

exec_kind = 'batch'
item_lvl_list = [5]
division_list = ['SELL_IN']    # SELL_IN / SELL_OUT

acc_classify_standard = 0.25

root_path = os.path.join('/', 'opt', 'DF', 'fcst')
save_path = os.path.join(root_path, 'analysis', 'accuracy', exec_kind)

exec_cfg = {
    'save_db_yn': True,
    'cycle_yn': True,
}

print(f"Start calculating accuracy: {exec_kind}")
pipe_acc = PipelineAccuracy(
    exec_kind=exec_kind,
    exec_cfg=exec_cfg,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
    acc_classify_standard=acc_classify_standard
)
pipe_acc.run()
print("Calculating accuracy is finished")
