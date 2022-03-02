import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineItem import PipelineItem

path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_cns': False,
    'cls_prep': True,
    'cls_train': True,
    'cls_pred': True,
    'cls_mdout': True
}

# Configuration
exec_cfg = {
    'cycle': True,                            # Prediction cycle

    # save configuration
    'save_step_yn': True,                    # Save each step result to object or csv
    'save_db_yn': False,                      # Save each step result to Database

    # Data preprocessing configuration
    'decompose_yn': False,                   # Decomposition
    'feature_selection_yn': False,           # Feature Selection
    'filter_threshold_cnt_yn': False,        # Filter data level under threshold count
    'filter_threshold_recent_yn': True,      # Filter data level under threshold recent week
    'filter_threshold_recent_sku_yn': True,  # Filter SKU level under threshold recent week
    'rm_fwd_zero_sales_yn': True,            # Remove forward empty sales
    'rm_outlier_yn': True,                   # Outlier clipping
    'data_imputation_yn': True,              # Data Imputation

    # Training configuration
    'scaling_yn': False,                     # Data scaling
    'grid_search_yn': False,                 # Grid Search
    'voting_yn': True                        # Add voting algorithm
}

print('------------------------------------------------')
print('Demand Forecast - SELL-IN')
print('------------------------------------------------')

# Data Configuration
data_cfg = {'division': 'SELL_IN'}

pipeline = PipelineItem(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    path_root=path_root
)

# Execute Baseline Forecast
pipeline.run()

print('------------------------------------------------')
print('Demand Forecast - SELL-OUT')
print('------------------------------------------------')

# Data Configuration
data_cfg = {'division': 'SELL_OUT'}

pipeline = PipelineItem(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    path_root=path_root
)

# Execute Baseline Forecast
pipeline.run()