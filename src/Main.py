import config
from DataPrep import DataPrep
from Model import ModelStats
from Model import ModelLstm


data_prep = DataPrep()

model_stats = ModelStats(sell_in=data_prep.sell_in_prep,
                         sell_out=data_prep.sell_out_prep)

best_models = model_stats.train()
model_stats.predict(model=best_models)

model_lstm = ModelLstm(sell_in=data_prep.sell_in_prep,
                       sell_out=data_prep.sell_out_prep)


print("")