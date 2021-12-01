import numpy as np

# Database Configuration
RDMS = 'mssql+pymssql'
HOST = '10.109.2.135'    # Database IP adress
DATABASE = 'BISCM'      # Database name
PORT = '1433'
USER = 'matrix'    # User name
PASSWORD = 'Diam0nd123!'     # User password

EXG_MAP = {
    '1202': '108',    # 노원 -> 서울
    '1005': '108',    # 도봉 -> 서울
    '1033': '159',    # 중앙 -> 부산
    '1212': '279',    # 구미 -> 구미
    '1065': '999',    # 이마트
    '1066': '999',    # 롯데마트
    '1067': '999',    # 홈플러스
    '1073': '999',    # 롯데슈퍼
    '1074': '999',    # GS유통
    '1075': '999',    # 홈플러스슈퍼
    '1076': '999',    # 이마트슈퍼
    '1173': '999'  # 7-11
}

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
        'damped_trend': [True],
        'remove_bias': [True],
        'seasonal_period': ['4'],
        'seasonal': ['add'],
        'trend': ['add'],
        'alpha': [0.1, 0.2, 0.3],
        'beta': [0.1, 0.2, 0.3],
        'gamma': [0.1, 0.2, 0.3],
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

# Rename columns
HRCHY_CD_TO_DB_CD_MAP = {
    'biz_cd': 'item_attr01_cd', 'line_cd': 'item_attr02_cd', 'brand_cd': 'item_attr03_cd',
    'item_cd': 'item_attr04_cd', 'biz_nm': 'item_attr01_nm', 'line_nm': 'item_attr02_nm',
    'brand_nm': 'item_attr03_nm', 'item_nm': 'item_attr04_nm'
}

COL_RENAME2 = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}

#
COL_ITEM = ['biz_cd', 'biz_nm', 'line_cd', 'line_nm', 'brand_cd', 'brand_nm',
            'item_cd', 'item_nm', 'sku_cd', 'sku_nm']
COL_CUST = ['cust_grp_cd', 'cust_grp_nm']