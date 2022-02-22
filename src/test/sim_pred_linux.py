import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from simulation.simulation.Simulate import Simulate

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'scaling_yn': False,     # Data scaling
    'grid_search_yn': False    # Grid Search
}


# simulation configuration
# data_version = '20180102-20210103'
# division_cd = 'SELL_IN'
# date_from = '20210104'
# date_to = '20210228'
# cust_grp_cd = '1005'
# item_cd = '5100081'
# discount = 0.2

# Execute Arguments List
# data_version / division_cd / apply_date_from / apply_date_to / cust_grp_cd / item_cd(SKU) / apply_discount

# Set System arguments
params = sys.argv[1]
param_parsed = params.split('/')

# split System arguments
data_version = param_parsed[0]
division_cd = param_parsed[1]
date_from = param_parsed[2]
date_to = param_parsed[3]
cust_grp_cd = param_parsed[4]
item_cd = param_parsed[5]
discount = float(param_parsed[6])

lag = 'w1'
date = {'date_from': date_from, 'date_to': date_to}

sim = Simulate(
    data_version=data_version,
    division_cd=division_cd,
    cust_grp_cd=cust_grp_cd,
    date=date,
    lag=lag,
    exec_cfg=exec_cfg,
    item_cd=item_cd,
    discount=discount
)
# Run simulation
result = sim.simulate()
sim.save_result(result=result)
print(result)
