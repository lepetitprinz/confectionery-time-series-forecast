# Class Configuration
CLS_LOAD = False
CLS_CNS = False
CLS_PREP = False
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
HRCHY_PROD = ['biz_cd', 'line_cd', 'brand_cd', 'item_cd', 'sku_cd']
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
