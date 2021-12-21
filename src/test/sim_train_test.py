from simulation.deployment.PipelineTest import PipelineTest

division = 'SELL_IN'
lag = 'w1'

# Configuration
exec_cfg = {
    'save_step_yn': True,               # Save object on local
    'save_db_yn': False,                # Save date on DB
    'scaling_yn': False,                # Data scaling
    'grid_search_yn': False,            # Grid Search
    'filter_threshold_week_yn': True    # Filter threshold week
}

# Step Configuration
step_cfg = {
    'cls_sim_load': False,    # Data Load
    'cls_sim_prep': False,     # Data Preprocessing
    'cls_sim_train': True     # Training
}

pipeline = PipelineTest(
    division=division,
    lag=lag,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg
)

pipeline.run()
