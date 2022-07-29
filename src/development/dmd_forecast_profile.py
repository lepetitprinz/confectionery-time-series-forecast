import os
from baseline.post_process.profile import Profile


hist_to = '20220501'
path = os.path.join('..', '..')

exec_cfg = {
    'batch': False,

    # Save Configuration
    'save_step_yn': True,
    'save_db_yn': False,
    #
}

apply_cfg = {
    'weeks': 26,
    'sales_threshold': 5,
    'cv_threshold': 0.5,
    'acc_threshold': 0.8,
}

profile = Profile(
    exec_cfg=exec_cfg,
    apply_cfg=apply_cfg,
    path=path,
    hist_to=hist_to,
)

# Run profiling
profile.run()
