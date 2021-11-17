from baseline.deployment.Pipeline import Pipeline

# Sales Data configuration
division = 'SELL_IN'

# Level Configuration
lvl_cfg = {
    'cust_lvl': 1,   # Customer group
    'item_lvl': 5,    # Biz - Line - Brand - Item - SKU
}
# Data IO Configuration
exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'decompose_yn': False,           # Decomposition
    'scaling_yn': False,             # Data scaling
    'impute_yn': True,               # Data Imputation
    'rm_outlier_yn': True,           # Outlier Correction
    'feature_selection_yn': False    # Feature Selection
}

# Execute Configuration
step_cfg = {
    'cls_load': False,
    'cls_cns': False,
    'cls_prep': True,
    'cls_train': True,
    'cls_pred': True,
    'cls_rpt': True
}

# Load result configuration
exec_rslt_cfg = {
    'train': False,
    'predict': False
}


pipeline = Pipeline(
    division=division,
    lvl_cfg=lvl_cfg,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    exec_rslt_cfg=exec_rslt_cfg
)

# Execute Baseline Forecast
pipeline.run()

