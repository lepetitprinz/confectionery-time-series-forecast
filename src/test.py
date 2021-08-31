import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from baseline.preprocess.DataPrep import DataPrep
from baseline.preprocess.ConsistencyCheck import ConsistencyCheck
from baseline.model.Train import Train
from baseline.model.Predict import Predict

import os
###################
# Directory Setting
###################
path_sell_in = os.path.join('..', 'data', 'sell_in_org.csv')
path_sell_in_cns = os.path.join('..', 'data', 'sell_in_check.csv')
path_sell_in_prep = os.path.join('..', 'result', 'sell_in_prep.pickle')
path_sell_in_score = os.path.join('..', 'result', 'sell_in_score.pickle')
path_sell_in_predict = os.path.join('..', 'result', 'sell_in_predict.pickle')

#######################
# SELL-IN
#######################

#######################
# 1. Load the dataset
#######################
data_io = DataIO()
common = data_io.get_dict_from_db(sql=SqlConfig.sql_comm_master(), key='OPTION_CD', val='OPTION_VAL')
date = {'date_from': common['rst_start_day'], 'date_to': common['rst_end_day']}
# sell_in = data_io.get_df_from_db(sql=SqlConfig.sql_sell_in(**date))
# data_io.save_object(data=sell_in, file_path=path_sell_in, kind='csv')  # Save object

# sell_in = data_io.load_object(file_path=path_sell_in, kind='csv')  # Load object

#######################
# 2. Consistency Check
#######################
# cns_check = ConsistencyCheck(division='sell_in', save_yn=False)

# sell_in_checked = cns_check.check(df=sell_in)
# data_io.save_object(data=sell_in_checked, file_path=path_sell_in_cns, kind='csv')
sell_in_checked = data_io.load_object(file_path=path_sell_in_cns, kind='csv')  # Load object

#######################
# 3. Data Preprocessing
#######################
prep = DataPrep()

#############################
# Biz - Line - Brand - Item
#############################
data_preped = prep.preprocess(data=sell_in_checked, division='SELL-IN')
# path_sell_in_4_prep = util.make_path(module='data', division='sell-in', hrchy_lvl=4, step='prep', data_type='pickle')
# data_io.save_object(data=data_preped, kind='binary', file_path=path_sell_in_4_prep)  # Save object
# data_preped = data_io.load_object(file_path=path_sell_in_4_prep, kind='binary')  # Load object

#####################
# Training
#####################
# Load from data
# cand_models = data_io.get_df_from_db(sql=SqlConfig.sql_algorithm(**{'division': 'FCST'}))
# cand_models = list(cand_models.T.iloc[0])
#
# param_grid = data_io.get_df_from_db(sql=SqlConfig.sql_best_hyper_param_grid())
# param_grid['stat'] = param_grid['stat'].apply(lambda x: x.lower())
# param_grid['option_cd'] = param_grid['option_cd'].apply(lambda x: x.lower())
# param_grid = util.make_lvl_key_val_map(df=param_grid, lvl='stat', key='option_cd', val='option_val')
#
# training = Train(division='SELL-IN',
#                  cand_models=cand_models,
#                  param_grid=param_grid,
#                  end_date=date['date_to'])
#
# # Train the model
# scores = training.train(df=data_preped)
# scores_db = training.make_score_result(scores=scores)
#
# # Save the score
# data_io.insert_to_db(df=scores_db, tb_name='M4S_I110410')

# save_dir = os.path.join('..', 'result', 'score_sell_in.pickle')
# data_io.save_object(data=scores, file_path=path_sell_in_score, kind='binary')  # Save object
# scores = data_io.load_object(file_path=path_sell_in_score, kind='binary')  # Load object

#############################
# Biz - Line - Brand
#############################
# data_lvl_3 = util.group(hrchy=config.HRCHY, hrchy_lvl=2, data=sell_in_checked)


# #####################
# # Prediction
# #####################
# pred = Predict()
# prediction = pred.forecast(df=data_preped)
# prediction_db = pred.make_pred_result(df=prediction)
# data_io.insert_to_db(df=prediction_db, tb_name='M4S_I110400')

# data_io.save_object(data=prediction, file_path=path_sell_in_predict, kind='binary')  # Save object
# prediction = data_io.load_object(file_path=path_sell_in_predict, kind='binary')  # Load object

# # Save the prediction
# model.save_prediction(predictions=prediction)
