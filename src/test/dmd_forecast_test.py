import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineReal import PipelineReal

# Sales Data configuration
division = 'SELL_IN'    # SELL_IN / SELL_OUT
cycle = 'w'    # SELL-OUT : w(week) / m(month)

path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
}

# Configuration
exec_cfg = {
    'cycle': True,                           # Prediction cycle
    # save configuration
    'save_step_yn': True,                   # Save each step result to object or csv
    'save_db_yn': False,                     # Save each step result to Database
    # Data preprocessing configuration
    'decompose_yn': False,                   # Decomposition
    'feature_selection_yn': False,           # Feature Selection
    'filter_threshold_cnt_yn': False,        # Filter data level under threshold count
    'filter_threshold_recent_yn': True,      # Filter data level under threshold recent week
    'filter_threshold_recent_sku_yn': True,  # Filter SKU level under threshold recent week
    'rm_fwd_zero_sales_yn': True,            # Remove forward empty sales
    'rm_outlier_yn': True,                   # Outlier Correction
    'data_imputation_yn': True,              # Data Imputation
    # Training configuration
    'scaling_yn': False,                     # Data scaling
    'grid_search_yn': False,                 # Grid Search
}

# Execute Configuration
step_cfg = {
    'cls_load': True,
    'cls_cns': True,
    'cls_prep': True,
    'cls_train': False,
    'cls_pred': False,
    'cls_mdout': False
}

# Load result configuration
exec_rslt_cfg = {
    'train': False,
    'predict': False,
    'middle_out': False
}

print('------------------------------------------------')
print(f'Demand Forecast - {division}')
print('------------------------------------------------')
# Check start time
print("Forecast Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = PipelineReal(
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