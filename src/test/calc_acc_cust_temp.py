import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccByCustomer import CalcAccByCustomer

root_path = os.path.join('/', 'opt', 'DF', 'fcst')
exec_kind = 'dev'
pred_exec_day = '20220131'    # W07(20220207) / W06(20220131)

# Business code: P1
biz_code = 'P1'
div_sp1_map = {
        # 'SELL_IN': ['1022', '1005', '1107', '1051', '1063', '1128', '1173'],
        'SELL_OUT': ['1065', '1067', '1066', '1076', '1075', '1073', '1074']
    }

acc_p1 = CalcAccByCustomer(
    biz_code=biz_code,
    div_sp1_map=div_sp1_map,
    root_path=root_path,
    cycle=False,
    pred_exec_day=pred_exec_day,
    exec_kind=exec_kind,
)
acc_p1.run()

# # Business code: P2
# biz_code = 'P2'
# div_sp1_map = {'SELL_IN': ['1098', '1017', '1112', '1206', '1213', '1128', '1101']}
#
# acc_p2 = CalcAccByCustomer(biz_code=biz_code, div_sp1_map=div_sp1_map, root_path=root_path)
# acc_p2.run()
