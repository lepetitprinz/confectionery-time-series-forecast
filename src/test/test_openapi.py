# from dao.OpenAPI import OpenAPI
# from dao.DataIO import DataIO
# from common.SqlConfig import SqlConfig
#
#
# sql_conf = SqlConfig()
#
# data_io = DataIO()
# info = data_io.get_dict_from_db(sql=sql_conf.sql_comm_master(),
#                                 key='OPTION_CD', val='OPTION_VAL')
#
# api = OpenAPI(info=info)
# data_list = api.get_api_dataset()
# for stn_data, stn_info in data_list:
#     for exg_data, exg_id in stn_data:
#         stn_info['idx_cd'] = exg_id
#         data_io.delete_from_db(sql_conf.del_openapi(**stn_info))
#         data_io.insert_to_db(df=exg_data, tb_name='M4S_O110710')


from dao.OpenAPIDust import OpenAPIDust
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig


sql_conf = SqlConfig()

data_io = DataIO()
info = data_io.get_dict_from_db(sql=sql_conf.sql_comm_master(),
                                key='OPTION_CD', val='OPTION_VAL')

api = OpenAPIDust(info=info)
data_list = api.get_api_dataset()
for stn_data, stn_info in data_list:
    for exg_data, exg_id in stn_data:
        exg_data = exg_data.fillna(0.0)
        stn_info['idx_cd'] = exg_id
        # data_io.delete_from_db(sql_conf.del_openapi(**stn_info))
        data_io.insert_to_db(df=exg_data, tb_name='M4S_O110710')
