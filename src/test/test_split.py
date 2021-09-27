import common.util as util
import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO
from baseline.tune.Split import Split

# Import Class
io = DataIO()
sql_conf = SqlConfig()

#  
save_steps_yn = False
load_step_yn = True

# Set data
data_vrsn_cd = '20210416-20210912'
division_cd = 'SELL_IN'

lvl_ratio = 6
lvl = {'lvl_ratio': lvl_ratio,
       'lvl_split': lvl_ratio-1}

fkey_ratio = config.LVL_FKEY_MAP[config.LVL_MAP[lvl['lvl_ratio']]]
fkey_split = config.LVL_FKEY_MAP[config.LVL_MAP[lvl['lvl_split']]]

args_ratio = {'data_vrsn_cd': data_vrsn_cd, 'division_cd': division_cd, 'fkey': fkey_ratio}
args_split = {'data_vrsn_cd': data_vrsn_cd, 'division_cd': division_cd, 'fkey': fkey_split}

df_ratio = io.get_df_from_db(sql=sql_conf.sql_pred_all(**args_ratio))
df_split = io.get_df_from_db(sql=sql_conf.sql_pred_all(**args_split))

file_path_ratio = util.make_path(module='object', division=division_cd, hrchy_lvl=str(lvl_ratio),
                                 step='_ratio', extension='pickle')
file_path_split = util.make_path(module='object', division=division_cd, hrchy_lvl=str(lvl_ratio),
                                 step='_split', extension='pickle')
# if save_steps_yn:
#     io.save_object(data=df_ratio, file_path=file_path_ratio, data_type='binary')
#     io.save_object(data=df_split, file_path=file_path_split, data_type='binary')
#
# if load_step_yn:
#     df_ratio = io.load_object(file_path=file_path_ratio, data_type='binary')
#     df_split = io.load_object(file_path=file_path_split, data_type='binary')

split = Split(data_vrsn_cd=args_ratio['data_vrsn_cd'],
              division_cd=args_ratio['division_cd'],
              lvl=lvl)

split.run(df_ratio=df_ratio, df_split=df_split)
