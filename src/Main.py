from model.Model import Model
from preprocess.DataPrep import DataPrep
from dao.SqlConfig import SqlConfig
from dao.SqlSession import SqlSession
from preprocess.ConsistencyCheck import ConsistencyCheck

import os
import pickle
import pandas as pd

# Connect to the DB
sql_conf = SqlConfig()
session = SqlSession()
session.init()

date_from = session.select(sql=sql_conf.get_comm_master(option='RST_START_DAY')).values[0][0]
date_to = session.select(sql=sql_conf.get_comm_master(option='RST_END_DAY')).values[0][0]
sell_in = session.select(sql=sql_conf.get_sell_in(date_from=date_from, date_to=date_to))
session.close()

# Temp
save_dir = os.path.join('..', 'result', 'test_sell_in.csv')
sell_in.to_csv(save_dir, index=False)
sell_in = pd.read_csv(save_dir)

# sell_out = session.select(sql=sql_config.get_sell_out(date_from=date_from, date_to=date_to))

# Consistency Check
# Sell-int
save_dir = os.path.join('..', 'result', 'check_sell_in.csv')
cns_check = ConsistencyCheck(division='sell_in', save_yn=False)
sell_in_checked = cns_check.check(df=sell_in)
sell_in_checked.to_csv(save_dir, index=False)

sell_in_checked = pd.read_csv(save_dir)

# Data Preprocessing
prep = DataPrep()

save_dir = os.path.join('..', 'result', 'prep_sell_in.pickle')
data_preped = prep.preprocess(data=sell_in_checked, division='SELL-IN')

with open(save_dir, 'wb') as handle:
    pickle.dump(data_preped, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_dir, 'rb') as handle:
    data_preped = pickle.load(handle)

# # Modeling
model = Model(division='SELL-IN')
# scores = model.train(df=data_preped)

# save_dir = os.path.join('..', 'result', 'score_sell_in.pickle')
# # with open(save_dir, 'wb') as handle:
# #     pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(save_dir, 'rb') as handle:
#     scores = pickle.load(handle)
#
# model.save_score(scores=scores)

# prediction = model.forecast(df=data_preped)

save_dir = os.path.join('..', 'result', 'prediction_sell_in.pickle')
# with open(save_dir, 'wb') as handle:
#     pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_dir, 'rb') as handle:
    predictions = pickle.load(handle)
#
model.save_prediction(predictions=predictions)
print('')