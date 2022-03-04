import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.DataLifeCycle import DataLifeCycle

rm_data_interval = 12
exec_cfg = {
    'delete_yn': True,
    'backup_yn': False
}

root_path = os.path.join('/', 'opt', 'DF', 'fcst')
dir_list = ['data', 'result', 'prediction']
exec_kind_list = ['batch', 'dev', 'verify']

path_and_dir_info = {
    'root_path': root_path,
    'backup_path': os.path.join(root_path, 'backup'),
    'module': dir_list,
    'exec_kind': exec_kind_list,
}

print('------------------------------------------------')
print('Data Life Cycle Management')
print('------------------------------------------------')

# Check start time
print("Forecast Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

dlc = DataLifeCycle(
    rm_data_interval=rm_data_interval,
    exec_cfg=exec_cfg,
    path_and_dir_info=path_and_dir_info,
)

dlc.run()

# Check end time
print("Forecast End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("")




