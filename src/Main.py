import config
from DataPrep import DataPrep
from Model import Model

# Data Preprocessing
data_prep = DataPrep()

model = Model(sell_in=data_prep.sell_prep,
              sell_out=data_prep.sell_out_prep)

# Results of All Model
if config.BEST_OR_ALL == "all":
    forecast = model.forecast_all()
    model.save_result_all(forecast=forecast)

# Results of best Model
elif config.BEST_OR_ALL == 'best':
    best_models = model.train()
    forecast = model.forecast(best_models=best_models)
    model.save_result(forecast=forecast, best_models=best_models)

print("Entire process is finished")