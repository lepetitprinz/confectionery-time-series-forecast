import os

# Path configuration
BASE_DIR = os.path.join('..', 'data')
SELL_IN_DIR = os.path.join(BASE_DIR, 'sales_sell_in.csv')
SELL_OUT_DIR = os.path.join(BASE_DIR, 'sales_sell_out.csv')
SAVE_DIR = os.path.join('..', 'result', 'forecast')

# Data configuration
COL_TARGET = 'amt'    # Target variable
CRT_TARGET_YN = True    # Correct target variable or not
COL_TOTAL = {'univ': ['dt', COL_TARGET],    # univ: datetime + target
             'multi': ['dt', COL_TARGET, 'sales'],
             'exg': ['dt', COL_TARGET, 'sales']}
COL_EXO = ['dc']    # Exogenous variables

# Outlier handling configuration
SMOOTH_YN = True    # True / False    Smoothing or not
SMOOTH_METHOD = 'quantile'    # quantile / sigma
SMOOTH_RATE = 0.05    # Quantile rate

# Datetime configuration
COL_DATETIME = 'dt'    # Datetime format column
RESAMPLE_RULE = ['W']    # D / W / M    Data resampling rule (Day, Week, Month)

# Result configuration
BEST_OR_ALL = 'best'    # all / best
VAR_TYPE = 'exg'    # univ / multi / exg
GROUP_TYPE = ['pd', 'cust', 'all']    # pd / cust / all

# MODEL configuration
TRAIN_RATE = 0.7    # Train / Test split rate
MODEL_CANDIDATES = {'univ': ['ar', 'arma', 'arima', 'ses', 'hw'],
                    'multi': ['var'],
                    'exg': ['lstm_vn']}

# Statistical Model Hyper-parameters
# Configurations of each model
N_TEST = 12    # prediction

LAG = {'D': 7, 'W': 1, 'M': 1}      # AR
SEASONAL = True                     # AR
TREND = 'ct'                        # AR / VARMAX / HW
TREND_ARMA = 'c'                    # ARMA
PERIOD = {'D': 7, 'W': 2, 'M': 2}   # AR / HW
FREQUENCY = None                    # ARMA / ARIMA
TWO_LVL_ORDER = {'D': (7, 0),       # ARMA / VARMAX
                 'W': (1, 0),
                 'M': (1, 0)}
THR_LVL_ORDER = {'D': (7, 0, 1),    # ARIMA
                 'W': (1, 0, 1),
                 'M': (1, 0, 1)}

INIT_METHOD = 'estimated'    # Simple Exponential Smoothing
SMOOTHING = 0.2              # Simple Exponential Smoothing
OPTIMIZED = True             # Simple Exponential Smoothing

TREND_HW = 'add'       # Holt-Winters
DAMPED_TREND = True    # Holt-Winters
SEASONAL_HW = 'add'    # Holt-Winters
USE_BOXCOX = None      # Holt-Winters
REMOVE_BIAS = True     # Holt-Winters

# Deep Learning Hyper-parameters
TIME_STEP = 12    # 4 weeks
LSTM_UNIT = 32
EPOCHS = 100
BATCH_SIZE = 32