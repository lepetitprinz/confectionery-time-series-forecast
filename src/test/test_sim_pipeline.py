from simulation.deployment.Pipeline import Pipeline

division = 'SELL_IN'
hrchy_lvl = 5
lag = 'w1'

# Configuration
exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'scaling_yn': False,     # Data scaling
    'grid_search_yn': False    # Grid Search
}

# Step Configuration
step_cfg = {
    'cls_sim_load': False,
    'cls_sim_prep': True,
    'cls_sim_train': False
}

#
exec_rslt_cfg = {

}

pipeline = Pipeline(
    division=division,
    hrchy_lvl=hrchy_lvl,
    lag=lag,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg,
    exec_rslt_cfg=exec_rslt_cfg
)

pipeline.run()
