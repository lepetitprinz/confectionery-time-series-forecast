from dao.DataIO import DataIO
from baseline.preprocess.DataPrep import DataPrep
from baseline.analysis.Decomposition import Decomposition
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Model import Model

import os
###################
# Directory Setting
###################
path_sell_in = os.path.join('..', 'result', 'test_sell_in.csv')
path_sell_in_cns = os.path.join('..', 'result', 'check_sell_in.csv')
path_sell_in_prep = os.path.join('..', 'result', 'prep_sell_in.pickle')
path_sell_in_score = os.path.join('..', 'result', 'score_sell_in.pickle')
path_sell_in_predict = os.path.join('..', 'result', 'predict_sell_in.pickle')

#####################
# Load the dataset
#####################
data_io = DataIO()
#
# common = data_io.get_comm_info()
# date_from = common['rst_start_day']
# date_to = common['rst_end_day']
# sell_in = data_io.get_sell_in(date_from=date_from, date_to=date_to)
#
# data_io.save_object(data=sell_in, file_path=path_sell_in, kind='csv')  # Save object
# sell_in = data_io.load_object(file_path=path_sell_in, kind='csv')  # Load object
#
# #####################
# # Consistency Check
# #####################
# # Sell-in
# cns_check = ConsistencyCheck(division='sell_in', save_yn=False)
#
# sell_in_checked = cns_check.check(df=sell_in)
# data_io.save_object(data=sell_in_checked, file_path=path_sell_in_cns, kind='csv')  # Save object
sell_in_checked = data_io.load_object(file_path=path_sell_in_cns, kind='csv')  # Load object

#####################
# Data Preprocessing
#####################
prep = DataPrep()

data_preped = prep.preprocess(data=sell_in_checked, division='SELL-IN')
# data_io.save_object(data=data_preped, file_path=path_sell_in_prep, kind='binary')  # Save object
# data_preped = data_io.load_object(file_path=path_sell_in_prep, kind='binary')  # Load object

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
