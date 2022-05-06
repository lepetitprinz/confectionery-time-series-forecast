from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig


io = DataIO()  # Data In/Out class
sql_conf = SqlConfig()  # DB Query class

sp1c_list = {'sp1c_list': ('101', '102')}

data = io.get_df_from_db(sql=sql_conf.sql_sp1_sp1c_test(**sp1c_list))
data_list = tuple(data['cust_grp_cd'].to_list())
print("")