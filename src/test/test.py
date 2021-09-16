import pandas as pd

from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

from collections import defaultdict

sql_conf = SqlConfig()
io = DataIO()
exg = io.get_df_from_db(sql=sql_conf.sql_exg_data())
print("")

exg_map = defaultdict(lambda: defaultdict(list))
for lvl1, lvl2, date, val in zip(exg['idx_dtl_cd'], exg['idx_cd'], exg['yymm'], exg['ref_val']):
    exg_map[lvl1][lvl2].append((date, val))

for key1, val1 in exg_map.items():
    for key2, val2 in val1.items():
        temp = pd.DataFrame(val2, columns=['yymmdd', 'val'])
        exg_map[key1][key2] = temp
