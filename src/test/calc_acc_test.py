import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccuracy import PipelineAccuracy

hist_to = '20211226'    # W10(20220227) / W09(20220220) / W08(20220213) / W07(20220206)
exec_kind = 'batch'
item_lvl_list = [5]
division_list = ['SELL_IN']    # SELL_IN / SELL_OUT

root_path = os.path.join('..', '..')
# root_path = os.path.join('/', 'opt', 'DF', 'fcst')
save_path = os.path.join(root_path, 'analysis', 'accuracy', exec_kind)

exec_cfg = {
    'save_db_yn': True,
    'cycle_yn': False,
    'calc_raw_yn': False,
    'calc_csv_yn': False,
    'calc_summary': False,
    'calc_db': True
}

pipe_acc = PipelineAccuracy(
    exec_kind=exec_kind,
    exec_cfg=exec_cfg,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
    hist_to=hist_to
)

print(f"Apply end date of history: {hist_to}")
pipe_acc.run()
print("Calculating accuracy is finished")
