import datetime

# hist_from = '20190128'    # W04
# hist_to = '20220123'      # W04
# hist_from = '20190204'    # W05
hist_to = '20220130'        # W05

# Change data type (string -> datetime)
hist_to_datetime = datetime.datetime.strptime(hist_to, '%Y%m%d')

# Add dates
hist_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=156) + datetime.timedelta(days=1)
compare_from = hist_to_datetime + datetime.timedelta(days=1)
compare_to = hist_to_datetime + datetime.timedelta(days=7)
md_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=13) + datetime.timedelta(days=1)

# Change data type (datetime -> string)
hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
compare_from = datetime.datetime.strftime(compare_from, '%Y%m%d')
compare_to = datetime.datetime.strftime(compare_to, '%Y%m%d')
md_from = datetime.datetime.strftime(md_from, '%Y%m%d')

print("")