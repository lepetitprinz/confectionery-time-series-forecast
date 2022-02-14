import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from baseline.analysis.CalcAccuracy import CalcAccuracy

# hist_from = '20190204'    # W05(20190204) / W04(20190128)
hist_to = '20220130'        # W05(20220130) / W04(20220123)

# Change data type (string -> datetime)
hist_to_datetime = datetime.datetime.strptime(hist_to, '%Y%m%d')

# Add dates
hist_from = datetime.datetime.strptime(hist_to, '%Y%m%d') - datetime.timedelta(weeks=156) + datetime.timedelta(days=1)
compare_from = hist_to_datetime + datetime.timedelta(days=8)
compare_to = hist_to_datetime + datetime.timedelta(days=14)
# compare_from = hist_to_datetime + datetime.timedelta(days=1)
# compare_to = hist_to_datetime + datetime.timedelta(days=7)

# Change data type (datetime -> string)
hist_from = datetime.datetime.strftime(hist_from, '%Y%m%d')
compare_from = datetime.datetime.strftime(compare_from, '%Y%m%d')
compare_to = datetime.datetime.strftime(compare_to, '%Y%m%d')

print(f"compare from: {compare_from}")
print(f"compare to: {compare_to}")