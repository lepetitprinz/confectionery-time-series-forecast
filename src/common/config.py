# Class Configuration
CLS_LOAD = False
CLS_CNS = False
CLS_PREP = False
CLS_TRAIN = False
CLS_PRED = True
CLS_SPLIT = False

# Database Configuration
RDMS = 'mssql+pymssql'
HOST = '10.109.16.49'    # Database IP adress
DATABASE = 'BISCM'      # Database name
PORT = '1433'
USER = 'matrix'    # User name
PASSWORD = 'Diam0nd123!'     # User password

# Database Configuration (temp)
# RDMS = 'mssql+pymssql'
# HOST = '10.112.33.101'    # Database IP adress
# DATABASE = 'BISCM'      # Database name
# PORT = '1433'
# USER = 'sa'    # User name
# PASSWORD = 'matrixadm'     # User password

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
COL_ITEM = ['biz_cd', 'biz_nm', 'line_cd', 'line_nm', 'brand_cd', 'brand_nm',
             'item_cd', 'item_nm', 'sku_cd', 'sku_nm']
COL_CUST = ['cust_grp_cd', 'cust_grp_nm']


LVL_CD_LIST = ['biz_cd', 'line_cd', 'brand_cd', 'item_cd', 'sku_cd', 'cust_grp_cd']
LVL_MAP = {1: 'biz_cd', 2: 'line_cd', 3: 'brand_cd', 4: 'item_cd', 5: 'sku_cd', 6: 'cust_grp_cd'}
LVL_FKEY_MAP = {'biz_cd': 'C0-P1', 'line_cd': 'C0-P2', 'brand_cd': 'C0-P3',
                 'item_cd': 'C0-P4', 'sku_cd': 'C0-P5', 'cust_grp_cd': 'C1-P5'}