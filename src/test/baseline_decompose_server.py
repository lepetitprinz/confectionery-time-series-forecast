import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineDecompose import PipelineDecompose

path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Data Configuration
data_cfg = {
    'division': 'SELL_IN',
    'cycle': 'w',
    'date': {
        'history': {
            'from': '20201228',
            'to': '20211226'
        },
        'middle_out': {
            'from': '20210927',
            'to': '20211226'
        },
        'evaluation': {
            'from': '20211227',
            'to': '20220327'
        }
    }
}

# Execute Configuration
exec_cfg = {
    'decompose_yn': True,            # Decomposition
    'cycle': False,                  # Weekly cycle
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': True,              # Save result on DB
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,
    'rm_not_exist_lvl_yn': False
}

# Load result configuration
exec_rslt_cfg = {'decompose': False}

start_time = time.time()

# Line Level
pipeline_line = PipelineDecompose(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    item_lvl=2,
    path_root=path_root
)
pipeline_line.run()
print("Time Series Decomposition(Line Level) is finished.\n")

# Brand Level
pipeline_brand = PipelineDecompose(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    item_lvl=3,
    path_root=path_root
)
pipeline_brand.run()
print("Time Series Decomposition(Brand Level) is finished.\n")

# Item Level
pipeline_item = PipelineDecompose(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    item_lvl=4,
    path_root=path_root
)
pipeline_item.run()
print("Time Series Decomposition(Item Level) is finished.\n")

end_time = time.time()
print('running time: ', end_time - start_time)
