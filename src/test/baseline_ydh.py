import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineReal import PipelineReal

# Root path
path_root = os.path.join('/', 'opt', 'DF', 'fcst')

# Sales Data configuration
division = 'SELL_IN'    # SELL_IN / SELL_OUT
cycle = 'w'    # SELL-OUT : w(week) / m(month)
item_lvl = 3  # 3: brand

test_vrsn_cd = 'TEST_0124_SELL_IN_3YEAR_BRAND'

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_cns': False,
    'cls_prep': True,
    'cls_train': True,
    'cls_pred': True,
    'cls_mdout': True,
    'cls_rpt': False
}

# Configuration
exec_cfg = {
    'cycle': False,                          # Prediction cycle
    # save configuration
    'save_step_yn': True,                  # Save each step result to object or csv
    'save_db_yn': False,                    # Save each step result to Database
    # Data preprocessing configuration
    'decompose_yn': False,                  # Decomposition
    'feature_selection_yn': False,          # Feature Selection
    'filter_threshold_cnt_yn': False,       # Filter data level under threshold count
    'filter_threshold_recent_yn': True,    # Filter data level under threshold recent week
    'rm_fwd_zero_sales_yn': True,           # Remove forward empty sales
    'rm_outlier_yn': True,                  # Outlier Correction
    'data_imputation_yn': True,             # Data Imputation
    # Training configuration
    'scaling_yn': False,                    # Data scaling
    'grid_search_yn': False,                # Grid Search
}


# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
    'item_lvl': item_lvl,
    'test_vrsn_cd': test_vrsn_cd,
    'date': {
        'history': {
            'from': '20190121', # 20190114
            'to': '20220116' # 20220109
        },
        'middle_out': {
            'from': '20211018', # 20211011
            'to': '20220116' #  20220109
        },
        'evaluation': {
            'from': '20220117', #  20220110
            'to': '20220123' #  20220116
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
# Execute Baseline Forecast
pipeline.run()

end_time = time.time()

print('running time: ', end_time - start_time)
