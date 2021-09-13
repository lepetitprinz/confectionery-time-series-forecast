# Class Configuration
CLS_LOAD = False
CLS_CNS = False
CLS_PREP = True
CLS_TRAIN = True
CLS_PRED = True

# Database Configuration
RDMS = 'mssql+pymssql'
HOST = '10.112.33.101'    # Database IP adress
DATABASE = 'BISCM'      # Database name
PORT = '1433'
USER = 'sa'    # User name
PASSWORD = 'matrixadm'     # User password

# Data configuration
HRCHY_CUST = []
UNIT_CD = ['BOX', 'EA ', 'BOL']

# Algorithm configuration
VALIDATION_METHOD = 'train_test'    # train_test / walk-walk_forward

# Model Hyper-parameters
# Deep Learning model Hyper-parameters
TIME_STEP = 12    # 4 weeks
LSTM_UNIT = 32
EPOCHS = 100
BATCH_SIZE = 32

# Open API Configuration
STN_LIST = [108]

# Rename columns
COL_RENAME1 = {'biz_cd': 'item_attr01_cd', 'line_cd': 'item_attr02_cd', 'brand_cd': 'item_attr03_cd',
               'item_cd': 'item_attr04_cd', 'biz_nm': 'item_attr01_nm', 'line_nm': 'item_attr02_nm',
               'brand_nm': 'item_attr03_nm', 'item_nm': 'item_attr04_nm'}
COL_RENAME2 = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}

#
COL_NAMES = ['biz_cd', 'biz_nm', 'line_cd', 'line_nm', 'brand_cd', 'brand_nm',
             'item_cd', 'item_nm', 'sku_cd', 'sku_nm']
