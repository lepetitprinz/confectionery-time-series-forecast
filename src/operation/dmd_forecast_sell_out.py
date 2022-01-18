import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineCycle import PipelineCycle

# Sales Data configuration
division = 'SELL_OUT'    # SELL_IN / SELL_OUT
cycle = 'w'    # SELL-OUT : w(week) / m(month)

path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
}

# Configuration
exec_cfg = {
    'cycle': True,
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': True,              # Save each step result to Database
    'rm_not_exist_lvl_yn': False,    # Remove not exist data level
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'decompose_yn': False,           # Decomposition
    'scaling_yn': False,             # Data scaling
    'feature_selection_yn': False,   # Feature Selection
    'grid_search_yn': False,         # Grid Search
    'filter_threshold_week_yn': False,
    'rm_fwd_zero_sales_yn': True
}

# Execute Configuration
step_cfg = {
    'cls_load': True,
    'cls_cns': True,
    'cls_prep': True,
    'cls_train': True,
    'cls_pred': True,
    'cls_mdout': True
}

# Load result configuration
exec_rslt_cfg = {
    'train': False,
    'predict': False,
    'middle_out': False
}

print('------------------------------------------------')
print('Demand Forecast - SELL-OUT')
print('------------------------------------------------')
# Check start time
print("Forecast Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = PipelineCycle(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    path_root=path_root
)

# Execute Baseline Forecast
pipeline.run()

# Check end time
print("Forecast End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))