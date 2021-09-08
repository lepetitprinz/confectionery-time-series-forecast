import common.util as util
import common.config as config
from dao.DataIO import DataIO
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck

import os
###################
# Directory Setting
###################
path_sell_in = os.path.join('..', 'result', 'sell_in_org.csv')
path_sell_in_cns = os.path.join('..', 'result', 'sell_in_check.csv')
path_sell_in_cns_bn = os.path.join('..', 'result', 'sell_in_check.pickle')
path_sell_in_prep = os.path.join('..', 'result', 'sell_in_prep.pickle')
path_sell_in_score = os.path.join('..', 'result', 'sell_in_score.pickle')
path_sell_in_predict = os.path.join('..', 'result', 'sell_in_predict.pickle')

#####################
# Load the dataset
#####################
data_io = DataIO()

# common = data_io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')
# date = {'date_from': common['rst_start_day'], 'date_to': common['rst_end_day']}
# sell_in = data_io.get_df_from_db(sql=SqlConfig.sql_sell_in(**date))
# data_io.save_object(data=sell_in, file_path=path_sell_in, kind='csv')  # Save object
# sell_out = data_io.get_df_from_db(sql=SqlConfig.sql_sell_out(**date))

sell_in = data_io.load_object(file_path=path_sell_in, data_type='csv')  # Load object

#####################
# Consistency Check
#####################
# Sell-in
cns_check = ConsistencyCheck(division='sell_in', save_yn=False)

sell_in_checked = cns_check.check(df=sell_in)
data_io.save_object(data=sell_in_checked, file_path=path_sell_in_cns, data_type='csv')
sell_in_checked = data_io.load_object(file_path=path_sell_in_cns, data_type='csv')  # Load object

# #####################
# # Data Preprocessing
# #####################
prep = DataPrep()
test = util.group(hrchy=config.HRCHY,
                  hrchy_lvl=2,
                  data=sell_in_checked)


data_preped = prep.preprocess(data=sell_in_checked, division='SELL-IN')
data_io.save_object(data=data_preped, file_path=path_sell_in_prep, data_type='binary')  # Save object
data_preped = data_io.load_object(file_path=path_sell_in_prep, data_type='binary')  # Load object

#####################
# Training
#####################
# model = Model(division='SELL-IN')
#
# # Train the model
# scores = model.train(df=data_preped)
#
# save_dir = os.path.join('..', 'result', 'score_sell_in.pickle')
# data_io.save_object(data=scores, file_path=path_sell_in_score, kind='binary')  # Save object
# scores = data_io.load_object(file_path=path_sell_in_score, kind='binary')  # Load object
#
# # Save the score
# model.save_score(scores=scores)
#
# #####################
# # Prediction
# #####################
# prediction = model.forecast(df=data_preped)
#
# data_io.save_object(data=prediction, file_path=path_sell_in_predict, kind='binary')  # Save object
# prediction = data_io.load_object(file_path=path_sell_in_predict, kind='binary')  # Load object
#
# # Save the prediction
# model.save_prediction(predictions=prediction)
