import os

###########################
# Database Configuration
###########################
RDMS = 'mssql+pymssql'
HOST = '10.112.33.101'    # Database IP adress
DATABASE = 'BISCM'      # Database name
PORT = '1433'
USER = 'sa'    # User name
PASSWORD = 'matrixadm'     # User password

###########################
# Path configuration
###########################
BASE_DIR = os.path.join('..', 'data')
SAVE_DIR = os.path.join('..', 'result')

###########################
# Open API Configuration
###########################
URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
SERVICE_KEY = 'UYRNns1wVRWz8MIyaMqUcL%2BHhIsbY0xjNyzRyvBNZRwh9zefraNj4lh9eBLgOw%2B2c8lBV%2Fh1SbzyNV96aO3DUw%3D%3D'
PAGE = 1
START_DATE = 20210101    # Start Date
END_DATE = 20210530      # Emd Date
STN_ID = 108    # Seoul
STN_LIST = [108]

###########################
# Data configuration
###########################
HRCHY_LIST = ['biz_cd', 'line_cd', 'brand_cd', 'item_ctgr_cd']
HRCHY = [(1, 'biz_cd'), (2, 'line_cd'), (3, 'brand_cd'), (4, 'item_ctgr_cd')]
HRCHY_LEVEL = len(HRCHY) - 1

#
TIME_TYPE = 'W'

# Outlier handling configuration
SMOOTH_YN = True    # Smoothing or not (True / False)
SMOOTH_METHOD = 'quantile'    # quantile / sigma
SMOOTH_RATE = 0.05    # Quantile rate

# Datetime configuration
COL_DATETIME = 'dt'    # Datetime format column
RESAMPLE_RULE = 'W'    # Data resampling rule (D / W / M)

###########################
# Model configuration
###########################
TRAIN_RATE = 0.7    # Train / Test split rate
# MODEL_CANDIDATES = {'univ': ['ar', 'arma', 'arima', 'ses', 'hw'],
#                     'multi': ['var'],
#                     'exg': ['lstm_vn']}

MODEL_CANDIDATES = {'univ': ['ar', 'arma', 'arima', 'hw'],
                    'multi': ['var'],
                    'exg': ['lstm_vn']}

#####################################
#  Model Hyper-parameters
#####################################
# Statistical model hyper-parameters
N_TEST = 4    # prediction

LAG = {'D': 7, 'W': 1, 'M': 1}      # AR
SEASONAL = False                     # AR
TREND = 'c'                        # AR / VARMAX / HW
TREND_ARMA = 'c'                    # ARMA
PERIOD = {'D': 7, 'W': 2, 'M': 2}   # AR / HW
FREQUENCY = None                    # ARMA / ARIMA
TWO_LVL_ORDER = {'D': (1, 0),       # ARMA / VARMAX
                 'W': (1, 0),
                 'M': (1, 0)}
THR_LVL_ORDER = {'D': (1, 0, 0),    # ARIMA
                 'W': (1, 0, 0),
                 'M': (1, 0, 0)}

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
