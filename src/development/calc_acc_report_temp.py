import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineAccReport import PipelineAccReport

# W07(20220206) / W08(20220213) / W09(20220220) / W10(20220227) / W11(20220306) /
# W12(20220313) / W13(20220320) / W14(20220327) / W15(20220403) / W16(20220410) /
# W17(20220417) / W18(20220424) / W19(2022)
hist_to = '20220424'
exec_kind = 'dev'    # dev / verify
item_lvl_list = [5]
division_list = ['SELL_IN']    # SELL_IN / SELL_OUT
acc_classifier_list = [0.5]    # Cover rate

root_path = os.path.join('..', '..')
save_path = os.path.join(root_path, 'analysis', 'accuracy', exec_kind)

exec_cfg = {
    'save_db_yn': False,
    'cycle_yn': False,
    'summary_add_cnt': False
}

pipe_acc = PipelineAccReport(
    exec_kind=exec_kind,
    exec_cfg=exec_cfg,
    root_path=root_path,
    save_path=save_path,
    division_list=division_list,
    item_lvl_list=item_lvl_list,
    hist_to=hist_to,
    acc_classifier_list=acc_classifier_list,
)

print(f"Apply end date of history: {hist_to}")
pipe_acc.run()
print("Calculating accuracy is finished")
