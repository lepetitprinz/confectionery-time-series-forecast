from baseline.deployment.PipelineBak import PipelineBak

# Sales Data configuration
division = 'SELL_OUT'    # SELL_IN / SELL_OUT / 7-11
cycle = 'w'    # w(week) / m(month)
test_vrsn_cd = 'TEST006_SKU_LVL'

# Data Configuration
data_cfg = {
    'division': division,
    'cycle': cycle,
    'test_vrsn_cd': test_vrsn_cd
}

# Level Configuration
lvl_cfg = {
    'cust_lvl': 1,   # SP1
    'item_lvl': 3,    # Biz - Line - Brand - Item - SKU
}
# Configuration
exec_cfg = {
    'save_step_yn': True,            # Save each step result to object or csv
    'save_db_yn': False,             #
    'rm_not_exist_lvl_yn': False,    # Remove not exist data level
    'decompose_yn': False,           # Decomposition
    'scaling_yn': False,             # Data scaling
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False,   # Feature Selection
    'grid_search_yn': False          # Grid Search
}

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_cns': False,
    'cls_prep': True,
    'cls_train': True,
    'cls_pred': False,
    'clss_mdout': False,
    'cls_rpt': False
}

# Load result configuration
exec_rslt_cfg = {
    'train': False,
    'predict': False,
    'middle_out': False
}

# Unit Test Option
unit_cfg = {
    'unit_test_yn': False,
    'cust_grp_cd': '1202',
    'item_cd': '5100000'
}

pipeline = PipelineBak(
    data_cfg=data_cfg,
    lvl_cfg=lvl_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    exec_rslt_cfg=exec_rslt_cfg,
    unit_cfg=unit_cfg,
    test_vrsn_cd=test_vrsn_cd
)

# Execute Baseline Forecast
pipeline.run()
