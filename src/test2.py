from dao.OpenAPI import OpenAPI
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

sql_conf = SqlConfig()

data_io = DataIO()
info = data_io.get_dict_from_db(sql=sql_conf.sql_comm_master(),
                                key='OPTION_CD', val='OPTION_VAL')

api = OpenAPI(info=info)
data_list = api.get_api_dataset()
for stn_data in data_list:
    for exg_data in stn_data:
        data_io.insert_to_db(df=exg_data, tb_name='M4S_O110710')

