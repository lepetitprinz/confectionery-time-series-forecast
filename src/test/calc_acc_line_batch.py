import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccBySp1 import CalcAccBySp1

root_path = os.path.join('..', '..')
# root_path = os.path.join('/', 'opt', 'DF', 'fcst')

# Business code: P1
biz_code = 'P1'
div_sp1_map = {
        'SELL_IN': ['1022', '1005', '1107', '1051', '1063', '1128', '1173'],
        'SELL_OUT': ['1065', '1073']
    }

acc_p1 = CalcAccBySp1(biz_code=biz_code, div_sp1_map=div_sp1_map, root_path=root_path)
acc_p1.run()

# Business code: P2
biz_code = 'P2'
div_sp1_map = {'SELL_IN': ['1098', '1017', '1112', '1206', '1213', '1128', '1101']}

acc_p2 = CalcAccBySp1(biz_code=biz_code, div_sp1_map=div_sp1_map, root_path=root_path)
acc_p2.run()
