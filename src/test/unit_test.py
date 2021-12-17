from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from operation.Cycle import Cycle

io = DataIO()
sql_conf = SqlConfig()
common = io.get_dict_from_db(
    sql=SqlConfig.sql_comm_master(),
    key='OPTION_CD',
    val='OPTION_VAL'
)

cycle = Cycle(common=common, rule='w')
cycle.calc_period()
print("")
