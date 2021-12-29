import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.deployment.PipelineDecompose import PipelineDecompose

# Data Configuration
data_cfg = {
    'division': 'SELL_IN',
    'in_out': 'out',
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
    'cycle': False,
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': False,             #
    'decompose_yn': True,            # Decomposition
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,
    'rm_not_exist_lvl_yn': False
}

# Load result configuration
exec_rslt_cfg = {'decompose': True}

# Brand Level
pipeline_brand = PipelineDecompose(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    item_lvl=3
)
pipeline_brand.run()

# Item Level
pipeline_item = PipelineDecompose(
    data_cfg=data_cfg,
    exec_cfg=exec_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    item_lvl=4
)
pipeline_item.run()
