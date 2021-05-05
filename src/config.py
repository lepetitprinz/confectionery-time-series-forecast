import os

# path
BASE_DIR = os.path.join('..', 'data')
SELL_IN_DIR = os.path.join(BASE_DIR, 'sales_sell_in.csv')
SELL_OUT_DIR = os.path.join(BASE_DIR, 'sales_sell_out.csv')

# Data Condition
VAR_TYPE = 'univ'    # 'univ' or 'multi
GROUP_TYPE = ['pd', 'cust', 'all']    # 'pd', 'cust', 'all'
RESAMPLE_RULE = ['D', 'W']    # Data resampling rule (Day, Week, Month)

# MODEL Candidates
MODEL_CANDIDATES = {'univ': ['ar', 'arma', 'arima', 'ses', 'hw'],
                    'multi': ['var', 'varma', 'varmax']}

# Statistical Model Hyper-paramters
# # Configurations of each model
N_TEST = 6      # prediction

LAG = 7                 # AR
SEASONAL = True         # AR
TREND = 'ct'            # AR / VARMAX / HW
TREND_ARMA = 'c'        # ARMA
PERIOD = 12             # AR / HW
FREQUENCY = None        # ARMA / ARIMA
TWO_LVL_ORDER = (1, 0)  # ARMA / VARMAX
THR_LVL_ORDER = (1, 0, 1)    # ARIMA

INIT_METHOD = 'estimated'
SMOOTHING = 0.2         # Simple Exponential Smoothing
OPTIMIZED = True        # Simple Exponential Smoothing

TREND_HW = 'add'        # Holt-Winters
DAMPED_TREND = True     # Holt-Winters
SEASONAL_MTD = 'add'    # Holt-Winters
USE_BOXCOX = None      # Holt-Winters
REMOVE_BIAS = True      # Holt-Winters

# Deep Learning Hyper-parameters
LSTM_UNIT = 32
EPOCHS = 10
BATCH_SIZE = 32
