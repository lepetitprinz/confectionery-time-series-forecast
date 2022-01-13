import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineReal import PipelineReal

# Root path
# path_root = os.path.join('/', 'opt', 'DF', 'fcst')
path_root = os.path.join('..', '..')

# Sales Data configuration
division = 'SELL_OUT'    # SELL_IN / SELL_OUT
cycle = 'w'    # SELL-OUT : w(week) / m(month)

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_cns': True,
    'cls_prep': False,
    'cls_train': False,
    'cls_pred': False,
    'cls_mdout': False,
    'cls_rpt': False
}

# Configuration
exec_cfg = {
    'cycle': False,
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': False,             # Save result on DB
    'rm_not_exist_lvl_yn': False,    # Remove not exist data level
    'decompose_yn': False,           # Decomposition
    'scaling_yn': False,             # Data scaling
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,   # Feature Selection
    'grid_search_yn': False,         # Grid Search
    'filter_threshold_week_yn': False,
    'rm_fwd_zero_sales_yn': True
}

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
    'date': {
        'history': {
            'from': '20190107',  # 20200928
            'to': '20220102'     # 20210926
        },
        'middle_out': {
            'from': '20210628',
            'to': '20210926'
        },
        'evaluation': {
            'from': '20210927',
            'to': '20211226'
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

# Execute Baseline Forecast
pipeline.run()

end_time = time.time()

print('running time: ', end_time - start_time)
