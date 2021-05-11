import config
from DataPrep import DataPrep
from Model import ModelStats

data_prep = DataPrep()

model_stats = ModelStats(sell_in=data_prep.sell_in_prep,
                         sell_out=data_prep.sell_out_prep)

# Results of All Model
if config.BEST_OR_ALL == "all":
    forecast = model_stats.forecast_all()
    model_stats.save_result_all(forecast=forecast)

#Results of best Model
elif config.BEST_OR_ALL == 'best':
    best_models = model_stats.train()
    forecast = model_stats.forecast(best_models=best_models)
    model_stats.save_result(forecast=forecast, best_models=best_models)

print("Entire process is finished")