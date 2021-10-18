from simulation.simulation.Simulate import Simulate

import pandas as pd

# simulation configuration
data_version = '20210101-20210530'
division_cd = 'sell_in'
hrchy_lvl = 4
hrchy_code = 'P11100101'

save_obj_yn = True
scaling_yn = True
save_db_yn = False

# simulation data
data = {'yymmdd': ['20210606', '20210613', '20210620', '20210627'],
        'discount': [0, 10, 5, 20]}
lag = 'w1'
discount = pd.DataFrame(data)

sim = Simulate(
    data_version=data_version,
    division_cd=division_cd,
    hrchy_lvl=hrchy_lvl,
    lag=lag,
    scaling_yn=scaling_yn,
    save_obj_yn=save_obj_yn
)

result = sim.simulate(discount=discount, hrchy_code=hrchy_code)
print(result)
