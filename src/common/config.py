import os

###########################
# Class Configuration
###########################
CLS_LOAD = False
CLS_CNS = True
CLS_PREP = False
CLS_TRAIN = False
CLS_PRED = False

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
HRCHY_CUST = []
HRCHY_PROD = ['biz_cd', 'line_cd', 'brand_cd', 'item_cd', 'sku_cd']
UNIT_CD = ['BOX', 'EA ', 'BOL']
RESAMPLE_RULE = 'w'

###########################
# Algorithm configuration
###########################
TRAIN_RATE = 0.8    # Train / Test split rate
MODEL_TO_VARIATE = {'ar': 'univ',
                    'arima': 'univ',
                    'hw': 'univ',
                    'var': 'multi',
                    'varmax': 'multi',
                    'sarima': 'multi',
                    'lstm': 'multi'}
VALIDATION_METHOD = 'train_test'    # train_test / walk-walk_forward

# Outlier handling configuration
SMOOTH_YN = True    # Smoothing or not (True / False)
SMOOTH_METHOD = 'quantile'    # quantile / sigma
SMOOTH_RATE = 0.05    # Quantile rate

#####################################
#  Model Hyper-parameters
#####################################
# Statistical model hyper-parameters
N_TEST = 4    # prediction

# Deep Learning model Hyper-parameters
TIME_STEP = 12    # 4 weeks
LSTM_UNIT = 32
EPOCHS = 100
BATCH_SIZE = 32
