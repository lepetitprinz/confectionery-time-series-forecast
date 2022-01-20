import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineReal import PipelineReal
import common.config as config

# Root path
path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Sales Data configuration
division = 'SELL_IN'    # SELL_IN / SELL_OUT
in_out = 'out'    # SELL-IN : out / in
cycle = 'w'    # SELL-OUT : w(week) / m(month)
item_lvl = 3 # 3: brand

test_vrsn_cd = 'TEST_0110_SELL_IN_3YEAR_BRAND'

# Execute Configuration
step_cfg = {
    'cls_load': True,
    'cls_cns': True,
    'cls_prep': False,
    'cls_train': False,
    'cls_pred': False,
    'cls_mdout': True,
    'cls_rpt': False
}

# Configuration
exec_cfg = {
    'cycle': False,
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': False,             #
    'rm_not_exist_lvl_yn': False,    # Remove not exist data level
    'decompose_yn': False,           # Decomposition
    'scaling_yn': False,             # Data scaling
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,   # Feature Selection
    'grid_search_yn': False,          # Grid Search
    'filter_threshold_cnt_yn': True,
    'rm_fwd_zero_sales_yn': True
}

# Data Configuration
data_cfg = {
    'division': division,
    'in_out': in_out,
    'cycle': cycle,
    'item_lvl': item_lvl,
    'test_vrsn_cd': test_vrsn_cd,
    'date': {
        'history': {
            'from': '20181001',    # 20201005
            'to': '20211003'    # 20211003
        },
        'middle_out': {
            'from': '20210705', # 20210705
            'to': '20211003' # 20211003
        },
        'evaluation': {
            'from': '20211004',    # 20210927
            'to': '20220102'    # 20211226
        }
    }
}

# Load result configuration
exec_rslt_cfg = {
    'train': False,
    'predict': False,
    'middle_out': False
}



pipeline = PipelineReal(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    path_root=path_root
)

start_time = time.time()
print(test_vrsn_cd)
print('threshold: ', config.threshold_cnt)
# Execute Baseline Forecast
pipeline.run()

end_time = time.time()

print('running time: ', end_time - start_time)
