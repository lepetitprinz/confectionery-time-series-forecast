from simulation.simulation.Simulate import Simulate


import sys

# simulation configuration
data_version = '20180102-20210103'
division_cd = 'SELL_IN'
item_cd = '5140889'

lag = 'w1'

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': True,
    'scaling_yn': False,     # Data scaling
    'grid_search_yn': False    # Grid Search
}
# simulation data

discount = 0.2
date_from = '20210105'
date_to = '20210212'

date = {'date_from': date_from, 'date_to': date_to}

sim = Simulate(
    data_version=data_version,
    division_cd=division_cd,
    date=date,
    lag=lag,
    exec_cfg=exec_cfg,
    item_cd=item_cd,
    discount=discount
)

result = sim.simulate()
sim.save_result(result=result)

print(result)

