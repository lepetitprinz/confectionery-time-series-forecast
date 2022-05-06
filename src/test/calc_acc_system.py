import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccSystem import PipelineAccSystem

exec_kind = 'batch'
item_lvl_list = [5]
division_list = ['SELL_IN']    # SELL_IN / SELL_OUT

# W07(20220206) / W08(20220213) / W09(20220220) / W10(20220227) / W11(20220306) / W12(20220313)
# W13(20220320) / W14(20220327) / W15(20220403) / W16(20220410) / W17(20220417)
hist_to = '20211226'

acc_classify_standard = 0.4

root_path = os.path.join('..', '..')
# root_path = os.path.join('/', 'opt', 'DF', 'fcst')
save_path = os.path.join(root_path, 'analysis', 'accuracy', exec_kind)

exec_cfg = {
    'save_file_yn': False,
    'save_db_yn': True,
    'cycle_yn': False,
}

pipe_acc = PipelineAccSystem(
    exec_kind=exec_kind,
    exec_cfg=exec_cfg,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
    acc_classify_standard=acc_classify_standard,
    hist_to=hist_to
)

pipe_acc.run()
print("Calculating accuracy is finished")
