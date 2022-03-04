import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.DataLifeCycle import DataLifeCycle

exec_cfg = {
    'delete_yn': False,
    'backup_yn': True
}

path_root = os.path.join('..', '..')
# path_root = os.path.join('/', 'opt', 'DF', 'fcst')
dir_list = ['data', 'result', 'prediction']
exec_kind_list = ['batch', 'dev', 'verify']

path_info = {
    'root_path': os.path.join('..', '..'),
    'backup_path': os.path.join('..', '..', 'backup'),
    'module': ['data', 'result', 'prediction'],
    'exec_kind': ['batch', 'dev', 'verify'],
}

dlc = DataLifeCycle(
    exec_cfg=exec_cfg,
    path_info=path_info,
)

dlc.run()



