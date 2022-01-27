import datetime

exec_date = '20220123'
exec_date = datetime.datetime.strptime(exec_date, '%Y%m%d') + datetime.timedelta(days=1)

sku_recent_date = '20220101'
sku_recent_date = datetime.datetime.strptime(sku_recent_date, '%Y%m%d')

if (exec_date - sku_recent_date) > 91:
    print("!!")