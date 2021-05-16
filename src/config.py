import os

###########################
# Path configuration
###########################
BASE_DIR = os.path.join('..', 'data')
SELL_IN_DIR = os.path.join(BASE_DIR, 'sales_sell_in.csv')
SELL_OUT_DIR = os.path.join(BASE_DIR, 'sales_sell_out.csv')
SAVE_DIR = os.path.join('..', 'result')

###########################
# Data configuration
###########################
# Scenario
# s1: univariate time series (datetime + target)
# s2: change target variable (datetime + revised target)
# s3: add "discount" variable (multivariate time series)
# s4: add "exogenous variable (add random noise)
SCENARIO = 's2'    # s1 / s2 / s3 / s4

# Result configuration
BEST_OR_ALL = 'best'    # all / best
VAR_TYPE = 'univ'    # univ / multi / exg

COL_TARGET = 'amt'    # Target variable
CRT_TARGET_YN = True    # Correct target variable or not
ADD_EXO_YN = False    # Exogenous variable (True / False)
COL_TOTAL = {'univ': ['dt', COL_TARGET],    # univ: datetime + target
             'multi': ['dt', COL_TARGET, 'sales'],
             'exg': ['dt', COL_TARGET, 'sales']}

# Product group configuration
PROD_GROUP = {'가': 'g1', '나': 'g1', '다': 'g2',
              '라': 'g2', '마': 'g3', '바': 'g3'}

# Outlier handling configuration
SMOOTH_YN = True    # Smoothing or not (True / False)
SMOOTH_METHOD = 'quantile'    # quantile / sigma
SMOOTH_RATE = 0.05    # Quantile rate

# Datetime configuration
COL_DATETIME = 'dt'    # Datetime format column
RESAMPLE_RULE = ['W']    # Data resampling rule (D / W / M)

###########################
# Model configuration
###########################
TRAIN_RATE = 0.7    # Train / Test split rate
MODEL_CANDIDATES = {'univ': ['ar', 'arma', 'arima', 'ses', 'hw'],
                    'multi': ['var'],
                    'exg': ['lstm_vn']}

#####################################
#  Model Hyper-parameters
#####################################
# Statistical model hyper-parameters
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

# Deep Learning model Hyper-parameters
TIME_STEP = 12    # 4 weeks
LSTM_UNIT = 32
EPOCHS = 100
BATCH_SIZE = 32
