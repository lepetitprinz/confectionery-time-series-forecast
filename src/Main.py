from DataPrep import DataPrep
from SqlConfig import SqlConfig
from SqlSession import SqlSession
from ConsistencyCheck import ConsistencyCheck

# Connect to the DB
sql_conf = SqlConfig()
session = SqlSession()
session.init()

date_from = session.select(sql=sql_conf.get_comm_master(option='RST_START_DAY')).values[0][0]
date_to = session.select(sql=sql_conf.get_comm_master(option='RST_END_DAY')).values[0][0]
sell_in = session.select(sql=sql_conf.get_sell_in(date_from=date_from, date_to=date_to))
# sell_out = session.select(sql=sql_config.get_sell_out(date_from=date_from, date_to=date_to))

# Consistency Check
# cns_check = ConsistencyCheck(data=sell_in, division='SELL-IN')

prep = DataPrep()
prep.preprocess(data=sell_in)
