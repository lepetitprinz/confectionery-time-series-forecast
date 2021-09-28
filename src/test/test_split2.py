import common.util as util
import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO
from baseline.tune.Split2 import Split2

# Import Class
io = DataIO()
sql_conf = SqlConfig()

#
save_steps_yn = False
load_step_yn = True

# Set data
data_vrsn_cd = '20210416-20210912'
division_cd = 'SELL_IN'

lvl_ratio = 5
lvl_split = 3
lvl = {'lvl_ratio': lvl_ratio, 'lvl_split': lvl_split}

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

split = Split2(data_vrsn_cd=args_ratio['data_vrsn_cd'],
               division_cd=args_ratio['division_cd'],
               lvl=lvl)

df_ratio_filtered = split.filter_col(df=df_ratio, kind='ratio')
df_agg_4 = split.group_by_agg(df=df_ratio_filtered, group_lvl=4)
df_agg_3 = split.group_by_agg(df=df_agg_4, group_lvl=3)
df_agg_4 = split.rename_col(df=df_agg_4, lvl='lower')
df_agg_3 = split.rename_col(df=df_agg_3, lvl='upper')

# Calculate ratio
ratio_4 = split.merge(left=df_agg_3, right=df_agg_4)
ratio_4['ratio'] = ratio_4['qty_lower'] / ratio_4['qty_upper']
ratio_4 = split.drop_qty(df=ratio_4)

# Split upper level qty using ratio of lower level
df_split_filtered = split.filter_col(df=df_split, kind='split')
df_split = split.split_qty(df_split_filtered, ratio_4)

print("")