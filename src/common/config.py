import numpy as np

# Class Configuration: What-IF Simulation
CLS_SIM_LOAD = True
CLS_SIM_PREP = True
CLS_SIM_TRAIN = True

# Database Configuration
RDMS = 'mssql+pymssql'
HOST = '10.109.2.135'    # Database IP adress
DATABASE = 'BISCM'      # Database name
PORT = '1433'
USER = 'matrix'    # User name
PASSWORD = 'Diam0nd123!'     # User password

# Database Configuration
# RDMS = 'mssql+pymssql'
# HOST = '10.109.16.49'    # Database IP adress
# DATABASE = 'BISCM'      # Database name
# PORT = '1433'
# USER = 'matrix'    # User name
# PASSWORD = 'Diam0nd123!'     # User password

# Algorithm configuration
VALIDATION_METHOD = 'train_test'    # train_test / walk-walk_forward

# Model Hyper-parameters

# 1.Time Series Forecast
PARAM_GRIDS_FCST = {
    'ar': {
        'lags': ['7', '14', '28'],
        'period': ['1', '7', '14'],
        'seasonal': [True, False],
        'trend': ['t', 'ct']
    },
    'hw': {
        'damped_trend': [True, False],
        'remove_bias': [True, False],
        'seasonal_period': ['1', '7', '14'],
        'seasonal': ['add'],
        'trend': ['add']
    },
    'var': {
        'trend': ['c', 't', 'ct'],
        'ic': [None, 'bic']
    }
}


# 2.What-If Simulation
PARAM_GRIDS_SIM = {
    'rf': {  # Random Forest
        'n_estimators': list(np.arange(100, 500, 100)),
        'criterion': ['squared_error'],
        'min_samples_split': list(np.arange(2, 6, 1)),  # minimum number of samples required to split inner node
        'min_samples_leaf': list(np.arange(1, 6, 1)),   # have the effect of smoothing the model
        'max_features': ['auto']
    },
    'gb': {  # Gradient Boost
        'n_estimators': list(np.arange(100, 500, 100)),
        'criterion': ['friedman_mse'],
        'min_samples_split': list(np.arange(2, 6, 1)),  # minimum number of samples required to split inner node
        'min_samples_leaf': list(np.arange(1, 6, 1)),   # have the effect of smoothing the model
        'max_features': ['auto']
    },
    'et': {  # Extremely Randomized Trees
        'n_estimators': list(np.arange(100, 500, 100)),
        'criterion': ['squared_error'],
        'min_samples_split': list(np.arange(2, 6, 1)),  # minimum number of samples required to split inner node
        'min_samples_leaf': list(np.arange(1, 6, 1)),   # have the effect of smoothing the model
        'max_features': ['auto']
    },
    'mlp': {  # Multi-layer Perceptron
        'units': [8, 16, 32],
        'batch_size': 32,
    }
}

PARAM_GRIDS_BEST = {
    'rf': {  # Random Forest
        'n_estimators': 100,
        'criterion': 'squared_error',
        'min_samples_split': 2,  # minimum number of samples required to split inner node
        'min_samples_leaf': 2,   # have the effect of smoothing the model
        'max_features': 'auto'
    },
    'gb': {  # Gradient Boost
        'n_estimators': 100,
        'criterion': ['friedman_mse'],
        'min_samples_split': 2,  # minimum number of samples required to split inner node
        'min_samples_leaf': 2,   # have the effect of smoothing the model
        'max_features': 'auto'
    },
    'et': {  # Extremely Randomized Trees
        'n_estimators': 100,
        'criterion': 'squared_error',
        'min_samples_split': 2,  # minimum number of samples required to split inner node
        'min_samples_leaf': 2,   # have the effect of smoothing the model
        'max_features': 'auto'
    },
    'mlp': {  # Multi-layer Perceptron
        'units': 8,
        'batch_size': 32,
    }
}

# Deep Learning model Hyper-parameters
LSTM_UNIT = 32
EPOCHS = 100
BATCH_SIZE = 32

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