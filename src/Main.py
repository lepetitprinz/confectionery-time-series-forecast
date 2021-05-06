import config
from DataPrep import DataPrep

from Model import ModelStats

data_prep = DataPrep()

model_stats = ModelStats(sell_in=data_prep.sell_in_prep,
                         sell_out=data_prep.sell_out_prep)

best_models = model_stats.train()
forecast = model_stats.forecast(best_models=best_models)
model_stats.save_result(forecast=forecast, best_models=best_models)

print("Entire process is finished")