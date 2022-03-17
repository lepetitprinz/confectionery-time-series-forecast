from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from operation.Cycle import Cycle

import pandas as pd

io = DataIO()
common = io.get_dict_from_db(    # common information dictionary
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

cycle = Cycle(common=common)
cycle.calc_period()
print("")