from DataPrep import DataPrep
from Model import ModelStats
from Model import ModelLstm

data_prep = DataPrep()

model_stats = ModelStats(sell_in=data_prep.df_sell_in,
                         sell_out=data_prep.df_sell_out)

model_lstm = ModelLstm(sell_in=data_prep.df_sell_in,
                       sell_out=data_prep.df_sell_out)


print("")