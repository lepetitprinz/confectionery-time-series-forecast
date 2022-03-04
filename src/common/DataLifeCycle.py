from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import os
import shutil
import datetime


class DataLifeCycle(object):
    def __init__(self, rm_data_interval: int, exec_cfg: dict, path_and_dir_info: dict):
        # Class Configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )
        self.rm_data_interval = rm_data_interval
        self.exec_cfg = exec_cfg
        self.path_and_dir_info = path_and_dir_info
        self.data_version = ''

    def init(self):
        self.set_data_version()

    def run(self):
        # Initialization
        self.init()

        # Delete previous history dataset
        if self.exec_cfg['delete_yn']:
            self.delete()
        elif self.exec_cfg['backup_yn']:
            self.backup()

    def set_data_version(self):
        # get monday of this week
        today = datetime.date.today()
        this_monday = today - datetime.timedelta(days=today.weekday())
        prev_monday = this_monday - datetime.timedelta(days=7 * self.rm_data_interval)

        date_from = prev_monday - datetime.timedelta(days=int(self.common['week_hist']) * 7)
        date_to = prev_monday - datetime.timedelta(days=1)

        date_from = date_from.strftime('%Y%m%d')
        date_to = date_to.strftime('%Y%m%d')
        self.data_version = date_from + '-' + date_to

    def delete(self) -> None:
        for directory in self.path_and_dir_info['module']:
            for exec_kind in self.path_and_dir_info['exec_kind']:
                dir_path = os.path.join(self.path_and_dir_info['root_path'], directory, exec_kind, self.data_version)
                if os.path.isdir(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Directory is deleted: {os.path.join(directory, exec_kind, self.data_version)}")
                    except OSError as e:
                        print(f"Error: {e.filename} - {e.strerror}.")

    def backup(self) -> None:
        root_path = self.path_and_dir_info['root_path']
        backup_path = self.path_and_dir_info['backup_path']

        for directory in self.path_and_dir_info['module']:
            for exec_kind in self.path_and_dir_info['exec_kind']:
                dir_path = os.path.join(directory, exec_kind, self.data_version)
                org_path = os.path.join(root_path, dir_path)
                target_path = os.path.join(backup_path, dir_path)
                if os.path.isdir(org_path):
                    shutil.copytree(org_path, target_path, dirs_exist_ok=True)
                    print(f"Directory is backup: {os.path.join(directory, exec_kind, self.data_version)}")