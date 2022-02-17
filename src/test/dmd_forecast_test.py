import os
import sys
import time
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineDev import PipelineDev

# Root path
# path_root = os.path.join('/', 'opt', 'DF', 'fcst')
path_root = os.path.join('..', '..')

# Sales Data configuration
division = 'SELL_OUT'    # SELL_IN / SELL_OUT
hist_to = '20220130'     # W05(20220130) / W04(20220123)

# Change data type (string -> datetime)
hist_to_datetime = datetime.datetime.strptime(hist_to, '%Y%m%d')

# Add dates
hist_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=156) + datetime.timedelta(days=1)
md_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=13) + datetime.timedelta(days=1)

# Change data type (datetime -> string)
hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
md_from = datetime.datetime.strftime(md_from, '%Y%m%d')

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
    'save_step_yn': False,                     # Save each step result to object or csv
    'save_db_yn': False,                      # Save each step result to Database

    # Data preprocessing configuration
    'decompose_yn': False,                    # Decomposition
    'feature_selection_yn': False,            # Feature Selection
    'filter_threshold_cnt_yn': False,         # Filter data level under threshold count
    'filter_threshold_recent_yn': True,       # Filter data level under threshold recent week
    'filter_threshold_recent_sku_yn': False,  # Filter SKU level under threshold recent week
    'rm_fwd_zero_sales_yn': True,             # Remove forward empty sales
    'rm_outlier_yn': True,                    # Outlier clipping
    'data_imputation_yn': True,               # Data Imputation

    # Feature engineering configuration
    'rolling_statistics_yn': False,            # Add features of rolling statistics
    'representative_sampling_yn': True,        # Add features of representative sampling

    # Training configuration
    'scaling_yn': False,                      # Data scaling
    'grid_search_yn': False,                  # Grid Search
    'voting_yn': True                         # Add voting algorithm
}

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': 'w',
    'date': {
        'history': {
            'from': hist_from,
            'to': hist_to
        },
        'middle_out': {
            'from': md_from,
            'to': hist_to
        }
    }
}

pipeline = PipelineDev(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    path_root=path_root
)

# Execute Baseline Forecast
pipeline.run()
