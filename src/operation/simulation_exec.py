import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from simulation.simulation.SimulateDB import SimulateDB

# simulation configuration
lag = 'w1'
path_root = os.path.join('/', 'opt', 'DF', 'fcst', 'simulation', 'model')

exec_cfg = {'save_db_yn': True}

sim = SimulateDB(
    exec_cfg=exec_cfg,
    path_root=path_root,
    lag=lag
)

result = sim.run()
