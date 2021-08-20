# from openapi import OpenAPI
#
# import os
# import pandas as pd
# from datetime import datetime
#
# start_date = 20210101    # Start Date
# end_date = 20210228      # Emd Date
# stn_id = 108    # Seoul
#
# open_api = OpenAPI(start_date=start_date, end_date=end_date, stn_id=stn_id)
# weather = open_api.get_api_dataset()
# date = pd.to_datetime(weather['date'])
# weather['date'] = date.apply(lambda x: x.strftime('%y%m%d'))
#
# weather.to_csv(os.path.join('..', 'result', 'openapi_weather.csv'), index=False)
# print("")

from SqlSession import SqlSession

sess = SqlSession()
sess.init()

sql = 'select * from M4S_I002030'
data = sess.select(sql=sql)
print("")