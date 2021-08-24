from DataPrep import DataPrep
from SqlConfig import SqlConfig
from SqlSession import SqlSession
from ConsistencyCheck import ConsistencyCheck

import os
import pandas as pd

# Connect to the DB
# sql_conf = SqlConfig()
# session = SqlSession()
# session.init()
#
# date_from = session.select(sql=sql_conf.get_comm_master(option='RST_START_DAY')).values[0][0]
# date_to = session.select(sql=sql_conf.get_comm_master(option='RST_END_DAY')).values[0][0]
# sell_in = session.select(sql=sql_conf.get_sell_in(date_from=date_from, date_to=date_to))
# session.close()

# Temp
save_dir = os.path.join('..', 'result', 'test_sell_in.csv')
# sell_in.to_csv(save_dir, index=False)
sell_in = pd.read_csv(save_dir)

# sell_out = session.select(sql=sql_config.get_sell_out(date_from=date_from, date_to=date_to))

# Consistency Check
# Sell-int
cns_check = ConsistencyCheck(division='sell_in')
cns_check.check(df=sell_in)
# session.close()



# Data Preprocessing
prep = DataPrep()
prep.preprocess(data=sell_in)
