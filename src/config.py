import os

# path
BASE_DIR = os.path.join('..', 'data')
SELL_IN_DIR = os.path.join(BASE_DIR, 'sales_sell_in.csv')
SELL_OUT_DIR = os.path.join(BASE_DIR, 'sales_sell_out.csv')

# Data Condition
VAR_TYPE = 'univ'    # 'univ' or 'multi
GROUP_TYPE = ['pd', 'cust', 'all']    # 'pd', 'cust', 'all'
RESAMPLE_RULE = ['D', 'W', 'M']    # Data resampling rule (Day, Week, Month)

# Deep Learning Hyper-parameter
LSTM_UNIT = 32
EPOCHS = 10
BATCH_SIZE = 32
