import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from simulation.deployment.PipelineReal import PipelineReal

lag = 'w1'
date = {'history': {
    'from': '20201228',
    'to': '20211226'}
}

# Root path
path_root = os.path.join('/', 'opt', 'DF', 'fcst')

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
    'cls_sim_load': True,    # Data Load
    'cls_sim_prep': True,     # Data Preprocessing
    'cls_sim_train': True     # Training
}

pipeline = PipelineReal(
    lag=lag,
    date=date,
    path_root=path_root,
    exec_cfg=exec_cfg,
    step_cfg=step_cfg
)

start_time = time.time()
pipeline.run()
end_time = time.time()
print('running time: ', end_time - start_time)
