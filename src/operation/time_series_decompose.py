import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineDecompCycle import PipelineDecompCycle

path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Data Configuration
data_cfg = {
    'division': 'SELL_IN',
    'cycle': 'w',
}

# Execute Configuration
exec_cfg = {
    'decompose_yn': True,            # Decomposition
    'cycle': True,                   # Weekly cycle
    'save_step_yn': True,           # Save each ste p result to object or csv
    'save_db_yn': True,              # Save result on DB
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,
    'rm_not_exist_lvl_yn': False
}

print('------------------------------------------------')
print('Time Series Decomposition')
print('------------------------------------------------')

# Check start time
print("Decomposition Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Line Level
pipeline_line = PipelineDecompCycle(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    item_lvl=2,
    path_root=path_root
)
pipeline_line.run()
print("Time Series Decomposition(Line Level) is finished.\n")

# Brand Level
pipeline_brand = PipelineDecompCycle(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    item_lvl=3,
    path_root=path_root
)
pipeline_brand.run()
print("Time Series Decomposition(Brand Level) is finished.\n")

# Item Level
pipeline_item = PipelineDecompCycle(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    item_lvl=4,
    path_root=path_root
)
pipeline_item.run()
print("Time Series Decomposition(Item Level) is finished.\n")

# Check end time
print("Decomposition End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




