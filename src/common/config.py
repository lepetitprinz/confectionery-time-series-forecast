# Database Configuration (Operation)
RDMS = 'mssql+pymssql'
HOST = '10.109.2.143'      # Ops Database IP address
# HOST = '10.109.6.62'     # Dev Database IP address
DATABASE = 'BISCM'         # Database name
PORT = '1433'
USER = 'matrix'            # User name
PASSWORD = 'Diam0nd123!'   # User password

# Data configuration
agg_avg = ['discount', 'gsr_sum', 'rhm_avg', 'temp_avg', 'temp_max', 'temp_min', 'num_work_day']
agg_sum = ['qty']

date_col = 'yymmdd'
db_hrchy_item_cd = ['item_attr01_cd', 'item_attr02_cd', 'item_attr03_cd', 'item_attr04_cd', 'item_cd']
db_hrchy_item_nm = ['item_attr01_nm', 'item_attr02_nm', 'item_attr03_nm', 'item_attr04_nm', 'item_nm']

drop_col = ['division_cd', 'seq', 'unit_price', 'unit_cd', 'from_dc_cd', 'create_date', 'week', 'cust_cd']
exg_fixe = 'discount'

filter_threshold_cnt = 8
filter_threshold_recent = 13
filter_threshold_sku_recent = 13

grid_search_option = 'best'    # best / each

hrchy_cust = 'cust_grp_cd'
hrchy_item = ['biz_cd', 'line_cd', 'brand_cd', 'item_cd', 'sku_cd']

outlier_sigma = 2

resample_rule = 'w'
target_col = 'qty'

unit_cd = ['box', 'ea', 'bol']
validation_method = 'train_test'
voting_method = 'mean'

# Exogenous Variable
web_page = 1
wea_service_key = 'UYRNns1wVRWz8MIyaMqUcL%2BHhIsbY0xjNyzRyvBNZRwh9zefraNj4lh9eBLgOw%2B2c8lBV%2Fh1SbzyNV96aO3DUw%3D%3D'
wea_url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'

# Middle-out option
weights = [0.162, 0.175, 0.174, 0.211, 0.278]    # weights
weight_top_n = 5

# Model Hyper-parameters
# 1.Time Series Forecast
PARAM_GRIDS_FCST = {
    'ar': {
        'lags': ['2', '3', '4'],
        'period': ['2', '3', '4'],
        'seasonal': [True],
        'trend': ['c', 't', 'ct']
    },
    'hw': {
        'damped_trend': [True, False],
        'remove_bias': [True],
        'seasonal_period': ['2', '3', '4'],
        'seasonal': ['add', 'mul'],
        'trend': ['add', 'mul'],
        'alpha': [0.1, 0.2, 0.3],
        'beta': [0.1, 0.2, 0.3],
        'gamma': [0.1, 0.2, 0.3],
    },
    'var': {
        'maxlags': [4],
        'trend': ['c', 'ct', 'ctt'],
        'ic': [None, 'aic', 'hqic', 'bic']
    }
}

# 2.What-If Simulation
PARAM_GRIDS_SIM = {
    'rf': {  # Random Forest
        'n_estimators': [200, 400, 600],
        'min_samples_split': [2, 5, 10],  # minimum number of samples required to split inner node
        'min_samples_leaf': [1, 2, 4],   # have the effect of smoothing the model
        # 'max_features': ['auto', 'sqrt'],
        'max_depth': [20, 40, 60],
        'bootstrap': [True, False],
    },
    'gb': {  # Gradient Boost
        'n_estimators': [200, 400, 600],
        'min_samples_split': [2, 5, 10],  # minimum number of samples required to split inner node
        'min_samples_leaf': [1, 2, 4],   # have the effect of smoothing the model
        # 'max_features': ['auto', 'sqrt'],
        'max_depth': [20, 40, 60],
    },
    'et': {  # Extremely Randomized Trees
        'n_estimators': [200, 400, 600],
        'min_samples_split': [2, 5, 10],  # minimum number of samples required to split inner node
        'min_samples_leaf': [1, 2, 4],   # have the effect of smoothing the model
        # 'max_features': ['auto', 'sqrt']
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

HRCHY_SKU_TO_DB_SKU_MAP = {'sku_cd': 'item_cd', 'sku_nm': 'item_nm'}
#
COL_CUST = ['cust_grp_cd', 'cust_grp_nm']
COL_ITEM = ['biz_cd', 'biz_nm', 'line_cd', 'line_nm', 'brand_cd', 'brand_nm',
            'item_cd', 'item_nm', 'sku_cd', 'sku_nm']

EXG_MAP = {
    '1005': '108',    # 도봉 -> 서울
    '1008': '108',    # 은평 -> 서울
    '1011': '108',    # 김포 -> 서울
    '1012': '112',    # 주안 -> 인천
    '1013': '112',    # 인천 -> 인천
    '1017': '108',    # 금천 -> 서울
    '1018': '108',    # 강동 -> 서울
    '1022': '108',    # 안양 -> 서울
    '1023': '119',    # 용인 -> 수원
    '1024': '119',    # 오산 -> 수원
    '1033': '159',    # 중앙 -> 부산
    '1041': '155',    # 진해 -> 창원
    '1044': '279',    # 김천 -> 구미
    '1045': '143',    # 경산 -> 대구
    '1046': '143',    # 황금 -> 대구
    '1050': '152',    # 울산 -> 울산
    '1051': '152',    # 동울산 -> 울산
    '1058': '146',    # 전주 -> 전주
    '1059': '140',    # 군산 -> 군산
    '1060': '133',    # 서대전 -> 대전
    '1061': '133',    # 서대전 -> 대전
    '1062': '177',    # 홍성 -> 홍성
    '1063': '236',    # 논산 -> 부여
    '1064': '184',    # 제주직할 -> 제주
    '1065': '999',    # 이마트 -> 전체
    '1066': '999',    # 롯데마트 -> 전체
    '1067': '999',    # 홈플러스 -> 전체
    '1068': '999',    # 킴스클럽 -> 전체
    '1069': '999',    # 하나로마트 -> 전체
    '1070': '999',    # 메가마트 -> 전체
    '1071': '999',    # 서원유통 -> 전체
    '1072': '999',    # 코스트코 -> 전체
    '1073': '999',    # 롯데슈퍼 -> 전체
    '1074': '999',    # GS유통 -> 전체
    '1075': '999',    # 홈플러스슈퍼 -> 전체
    '1076': '999',    # 이마트슈퍼 -> 전체
    '1077': '999',    # 롯데백화점 -> 전체
    '1078': '999',    # 신세계백화점 -> 전체
    '1079': '999',    # LOBS -> 전체
    '1080': '999',    # 올리브영 -> 전체
    '1081': '999',    # GS왓슨스 -> 전체
    '1088': '999',    # 코레일 -> 전체
    '1089': '999',    # 특판 -> 전체
    '1090': '999',    # 쿠팡 -> 전체
    '1091': '999',    # 마켓컬리 -> 전체
    '1092': '108',    # 북서울 -> 서울
    '1093': '98',     # 의정부 -> 동두천
    '1094': '108',    # 남양주 -> 서울
    '1095': '108',    # 종로 -> 서울
    '1097': '108',    # 성수 -> 서울
    '1098': '99',     # 고양 -> 파주
    '1100': '101',    # 춘천 -> 춘천
    '1101': '112',    # 부천 -> 인천
    '1105': '119',    # 안산 -> 수원
    '1106': '119',    # 성남 -> 수원
    '1107': '108',    # 동작 -> 서울
    '1110': '108',    # 광명 -> 서울
    '1111': '999',    # 경원 -> ???
    '1112': '119',    # 수원 -> 수원
    '1116': '131',    # 청주 -> 청주
    '1117': '114',    # 원주 -> 원주
    '1118': '127',    # 충주 -> 충주
    '1119': '105',    # 강릉 -> 강릉
    '1120': '232',    # 천안 -> 천안
    '1121': '119',    # 평택 -> 수원
    '1122': '159',    # 부산건과 -> 부산
    '1123': '257',    # 양산 -> 양산
    '1125': '159',    # 부산진 -> 부산
    '1126': '159',    # 해운대 -> 부산
    '1127': '155',    # 마산 -> 창원
    '1128': '192',    # 진주 -> 진주
    '1129': '155',    # 창원 -> 창원
    '1130': '162',    # 통영 -> 통영
    '1131': '253',    # 김해 -> 김해
    '1133': '143',    # 대구 -> 대구
    '1134': '143',    # 서대구 -> 대구
    '1138': '143',    # 서부 -> 대구
    '1139': '138',    # 포항 -> 포항
    '1140': '136',    # 안동 -> 안동
    '1143': '999',    # 중부1 -> ???
    '1144': '999',    # 중부2 -> ???
    '1145': '156',    # 남광주 -> 광주
    '1146': '165',    # 목포 -> 목포
    '1147': '174',    # 순천 -> 순천
    '1148': '156',    # 서광주 -> 광주
    '1173': '999',    # 7-11
    '1174': '999',    # GS25
    '1175': '999',    # CU
    '1176': '999',    # 미니스톱
    '1177': '999',    # 이마트24
    '1178': '999',    # 씨스페이스
    '1196': '999',    # 디씽
    '1197': '999',    # 푸드조이
    '1198': '999',    # 보담
    '1200': '999',    # 배달의 민족
    '1201': '108',    # 미금 -> 서울
    '1202': '108',    # 노원 -> 서울
    '1203': '108',    # 마포 -> 서울
    '1204': '108',    # 양재 -> 서울
    '1205': '108',    # 영동 -> 서울
    '1206': '159',    # 새동래 -> 부산
    '1207': '159',    # 문현 -> 부산
    '1208': '159',    # 가야 -> 부산
    '1209': '184',    # 제주 -> 제주
    '1210': '143',    # 성서 -> 대구
    '1211': '143',    # 수성 -> 대구
    '1212': '279',    # 구미 -> 구미
    '1213': '143',    # 대구 -> 대구
    '1214': '156',    # 광주 -> 광주
    '1215': '174',    # 광양 -> 순천
    '1216': '140',    # 익산 -> 군산
    '1217': '999',    # 대리점 -> 전체
    '1218': '108',    # 강동 -> 서울
    '1219': '999',    # 중부 -> ???
    '1220': '108',    # 강서 -> 서울
    '1221': '108',    # 강남 -> 서울
    '1223': '119',    # 수원 -> 수원
    '1224': '105',    # 강원 -> 강릉
    '1225': '159',    # 부산 빙과 -> 부산
    '1226': '138',    # 경북 -> 포항
    '1227': '253',    # 경남 -> 김해
    '1228': '131',    # 충청 -> 청주
    '1229': '156',    # 호남 빙과 -> 광주
    '1230': '999',    # 나뚜루유통영업 -> 전국
    '1232': '999',    # 자사몰 -> 전국
    '1233': '108',    # 남서울 대리점 -> 서울
    '1234': '112',    # 도매전담영업소(부천) -> 인천
    '1235': '98',    # 도매전담영업소(양주) -> 동두천
    '1236': '108',    # 도매전담영업소(강동) -> 서울
    '1237': '108',    # 도매전담영업소(고양) -> 서울
}