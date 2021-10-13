import common.config as config

from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from simulation.model.Predict import Predict

division = 'sell_in'
hrchy_lvl = 4

save_obj_yn = True
load_obj_yn = True
save_db_yn = False

pred = Predict(
    data_version='',
    hrchy_lvl=hrchy_lvl
)