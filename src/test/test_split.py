import common.util as util
import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO
from baseline.tune.Split import Split

import os

# Import Class
io = DataIO()
sql_conf = SqlConfig()

#
save_steps_yn = False
load_step_yn = True

# Data configuration
data_vrsn_cd = '20210416-20210912'
division_cd = 'SELL_IN'

lvl_ratio = 5
lvl_split = 3
lvl = {'lvl_ratio': lvl_ratio, 'lvl_split': lvl_split}
hrchy_list = config.LVL_CD_LIST

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

split = Split(data_vrsn_cd=args_ratio['data_vrsn_cd'],
              division_cd=args_ratio['division_cd'],
              lvl=lvl)

result = split.run(df_split=df_split, df_ratio=df_ratio)
save_path = os.path.join('..', '..', 'tune', division_cd + '_' +
                         hrchy_list[lvl_split-1] + '_' + hrchy_list[lvl_ratio-1] + '.csv')
io.save_object(data=result, data_type='csv', file_path=save_path)
