import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineReal import PipelineReal

# Root path
path_root = os.path.join('/', 'opt', 'DF', 'fcst')
# path_root = os.path.join('..', '..')

# Sales Data configuration
division = 'SELL_IN'    # SELL_IN / SELL_OUT
cycle = 'w'    # SELL-OUT : w(week) / m(month)

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
    'cycle': False,                           # Prediction cycle

    # save configuration
    'save_step_yn': True,                    # Save each step result to object or csv
    'save_db_yn': False,                     # Save each step result to Database

    # Data preprocessing configuration
    'decompose_yn': False,                    # Decomposition
    'feature_selection_yn': False,            # Feature Selection
    'filter_threshold_cnt_yn': False,         # Filter data level under threshold count
    'filter_threshold_recent_yn': True,       # Filter data level under threshold recent week
    'filter_threshold_recent_sku_yn': False,  # Filter SKU level under threshold recent week
    'rm_fwd_zero_sales_yn': True,             # Remove forward empty sales
    'rolling_statistics_yn': False,           # Rolling Statistics
    'rm_outlier_yn': True,                    # Outlier Correction
    'data_imputation_yn': True,               # Data Imputation

    # Training configuration
    'scaling_yn': False,                      # Data scaling
    'grid_search_yn': False,                  # Grid Search
}

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
    'date': {
        'history': {
            'from': '20190128',  # 20190204
            'to': '20220123'     # 20220130
        },
        'middle_out': {
            'from': '20211025',  # 20211101
            'to': '20220123'     # 20220130
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
