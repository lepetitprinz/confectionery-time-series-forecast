import pandas as pd

from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

from collections import defaultdict

sql_conf = SqlConfig()
io = DataIO()
io.delete_from_db(sql=sql_conf.del_test())
