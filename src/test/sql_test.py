from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

io = DataIO()
sql_config = SqlConfig()

date = {'from': '20220103', 'to': '20220109'}

result = io.get_df_from_db(sql=sql_config.sql_cust_sp1_map_error(**date))

print("")