import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO
from baseline.tune.Split import Split

io = DataIO()
sql_conf = SqlConfig()

data_vrsn_cd = '20210416-20210912'
division_cd = 'SELL_IN'
lvl_ratio = 4

cd_ratio = config.LVL_MAP[lvl_ratio]
cd_split = config.LVL_MAP[lvl_ratio-1]
fkey_ratio = config.LVL_FKEY_MAP[cd_ratio]
fkey_split = config.LVL_FKEY_MAP[cd_ratio]

args_ratio = {'data_vrsn_cd': data_vrsn_cd, 'division_cd': division_cd, 'fkey': fkey_ratio}
args_split = {'data_vrsn_cd': data_vrsn_cd, 'division_cd': division_cd, 'fkey': fkey_split}

df_ratio = io.get_df_from_db(sql=sql_conf.sql_pred_all(**args_ratio))
df_split = io.get_df_from_db(sql=sql_conf.sql_pred_all(**args_split))

split = Split(data_vrsn_cd=args_ratio['data_vrsn_cd'],
              division_cd=args_ratio['division_cd'],
              rating_lvl=lvl_ratio)

split.run(df_ratio=df_ratio, df_split=df_split)
