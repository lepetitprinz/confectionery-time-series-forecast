import pandas as pd

import config
from DataPrep import DataPrep
from Model import Model
from sqlSession import SqlSession

from sqlalchemy import Table, MetaData, Column, String, Integer, Float, Date

sess = SqlSession()
sess.init()

# Query Test
# sql = 'select * from M4S_O110800'
# test = sess.select(sql=sql)
# print(test)

# Insert Test
table_name = "M4S_O110800"
table_meta = sess.get_table_info(tb_name=table_name)

sql_ist = f'INSERT INTO {table_name} ' \
          '(PROJECT_CD, DIVISION, MONTH, FKEY, RESULT_SALES, SEASON_VAL, TREND_VAL, RANDOM_VAL)' \
          ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)'

test_df = pd.DataFrame({'PROJECT_CD': ['ENT009', 'ENT009'],
                        'DIVISION': ['SELL-IN', 'SELL-IN'],
                        'MONTH': [1, 2],
                        'FKEY': [1, 2],
                        'RESULT_SALES': [100, 200],
                        'SEASON_VAL': [1, 2],
                        'TREND_VAL': [1, 2],
                        'RANDOM_VAL': [1, 2],
                        })

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