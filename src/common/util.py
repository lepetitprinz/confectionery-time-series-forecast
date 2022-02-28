import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import date, timedelta
from collections import defaultdict


def counting(hrchy_lvl, df=None, val=None, lvl=0, cnt=0):
    if lvl == 0:
        for key, val in df.items():
            cnt = counting(hrchy_lvl=hrchy_lvl, val=val, lvl=lvl+1, cnt=cnt)

    elif lvl < hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            cnt = counting(hrchy_lvl=hrchy_lvl, val=val_hrchy, lvl=lvl+1, cnt=cnt)

        return cnt

    elif lvl == hrchy_lvl:
        for key_hrchy in val.keys():
            cnt += 1
        return cnt

    return cnt


def group(data, hrchy, hrchy_lvl, cd=None, lvl=0, cnt=0) -> tuple:
    grp = {}
    col = hrchy[lvl]

    code_list = None
    if isinstance(data, pd.DataFrame):
        code_list = list(data[col].unique())

    elif isinstance(data, dict):
        code_list = list(data[cd][col].unique())

    if lvl < hrchy_lvl:
        for code in code_list:
            sliced = None
            if isinstance(data, pd.DataFrame):
                sliced = data[data[col] == code]
            elif isinstance(data, dict):
                sliced = data[cd][data[cd][col] == code]
            result, cnt = group(hrchy=hrchy, hrchy_lvl=hrchy_lvl, data={code: sliced},
                                cd=code, lvl=lvl + 1, cnt=cnt)
            grp[code] = result

    elif lvl == hrchy_lvl:
        temp = {}
        for code in code_list:
            cnt += 1
            sliced = None
            if isinstance(data, pd.DataFrame):
                sliced = data[data[col] == code]
            elif isinstance(data, dict):
                sliced = data[cd][data[cd][col] == code]
            temp[code] = sliced

        return temp, cnt

    return grp, cnt


def hrchy_recursion(hrchy_lvl, fn=None, df=None, val=None, lvl=0):
    temp = None
    if lvl == 0:
        temp = {}
        for key, val in df.items():
            result = hrchy_recursion(hrchy_lvl=hrchy_lvl, fn=fn, val=val, lvl=lvl + 1)
            temp[key] = result

    elif lvl < hrchy_lvl:
        temp = {}
        for key_hrchy, val_hrchy in val.items():
            result = hrchy_recursion(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy, lvl=lvl + 1)
            temp[key_hrchy] = result

        return temp

    elif lvl == hrchy_lvl:
        temp = {}
        for key_hrchy, val_hrchy in val.items():
            result = fn(val_hrchy)
            temp[key_hrchy] = result
        return temp

    return temp


def hrchy_recursion_with_none(hrchy_lvl, fn=None, df=None, val=None, lvl=0):
    temp = None
    if lvl == 0:
        temp = {}
        for key, val in df.items():
            result = hrchy_recursion_with_none(hrchy_lvl=hrchy_lvl, fn=fn, val=val, lvl=lvl + 1)
            if result is not None:
                temp[key] = result
        if len(temp) == 0:
            return None

    elif lvl < hrchy_lvl:
        temp = {}
        for key_hrchy, val_hrchy in val.items():
            result = hrchy_recursion_with_none(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy, lvl=lvl + 1)
            if result is not None:
                temp[key_hrchy] = result
        if len(temp) == 0:
            return None
        return temp

    elif lvl == hrchy_lvl:
        temp = {}
        for key_hrchy, val_hrchy in val.items():
            result = fn(val_hrchy)
            if result is not None:
                temp[key_hrchy] = result
        if len(temp) == 0:
            return None
        return temp

    return temp


def hrchy_recursion_with_key(hrchy_lvl, fn=None, df=None, val=None, lvl=0, hrchy=[]) -> None:
    if lvl == 0:
        for key, val in df.items():
            hrchy.append(key)
            hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val, lvl=lvl+1, hrchy=hrchy)
            hrchy.remove(key)

    elif lvl < hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            hrchy.append(key_hrchy)
            hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy, lvl=lvl+1, hrchy=hrchy)
            hrchy.remove(key_hrchy)

    elif lvl == hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            hrchy.append(key_hrchy)
            fn(hrchy, val_hrchy)
            hrchy.remove(key_hrchy)


def hrchy_recursion_extend_key(hrchy_lvl, fn=None, df=None, val=None, lvl=0, hrchy=[]):
    if lvl == 0:
        temp = []
        for key, val in df.items():
            hrchy.append(key)
            result = hrchy_recursion_extend_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val,
                                                lvl=lvl+1, hrchy=hrchy)
            temp.extend(result)
            hrchy.remove(key)

    elif lvl < hrchy_lvl:
        temp = []
        for key_hrchy, val_hrchy in val.items():
            hrchy.append(key_hrchy)
            result = hrchy_recursion_extend_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy,
                                                lvl=lvl+1, hrchy=hrchy)
            temp.extend(result)
            hrchy.remove(key_hrchy)

        return temp

    elif lvl == hrchy_lvl:
        temp = []
        for key_hrchy, val_hrchy in val.items():
            hrchy.append(key_hrchy)
            result = fn(hrchy, val_hrchy)
            temp.extend(result)
            hrchy.remove(key_hrchy)

        return temp

    return temp


def make_path_baseline(path: str, module: str, exec_kind: str, division: str, data_vrsn: str, hrchy_lvl: str,
                       step: str, extension: str):
    path_dir = os.path.join(path, module, exec_kind, data_vrsn)
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_file = os.path.join(path_dir, division + '_' + data_vrsn + '_' + str(hrchy_lvl) + step + '.' + extension)

    return path_file


def make_path_sim(path: str, module: str, division: str, data_vrsn: str, step: str, extension: str):
    path = os.path.join(path, 'simulation', module, division + '_' + data_vrsn + '_' + step + '.' + extension)

    return path


# def make_lvl_key_val_map(df: pd.DataFrame, lvl: str, key: str, val: str):
#     result = {}
#     for i in df[lvl].unique():
#         temp = df[df[lvl] == i]
#         result[i] = {}
#         for k, v in zip(temp[key], temp[val]):
#             result[i][k] = v
#
#     return result


def make_lvl_key_val_map(df: pd.DataFrame, lvl: str, key: str, val: str):
    result = defaultdict(lambda: defaultdict(dict))
    for lvl, key, val in zip(df[lvl], df[key], df[val]):
        result[lvl][key] = val

    return result


def prep_exg_all(data: pd.DataFrame):
    exg_map = defaultdict(lambda: defaultdict(list))
    for lvl1, lvl2, date, val in zip(data['idx_dtl_cd'], data['idx_cd'], data['yymm'], data['ref_val']):
        exg_map[lvl1][lvl2].append((date, val))

    weather_map = {}
    for location, val1 in exg_map.items():
        weather = pd.DataFrame()
        for key2, val2 in val1.items():
            temp = pd.DataFrame(val2, columns=['yymmdd', key2])
            temp = temp.sort_values(by='yymmdd')
            if len(weather) == 0:
                weather = pd.concat([weather, temp], axis=1, join='outer')
            else:
                weather = pd.merge(weather, temp, on='yymmdd')

        weather.columns = [col.lower() for col in weather.columns]
        weather.loc[:, 'yymmdd'] = weather.loc[:, 'yymmdd'].astype(np.int64)
        weather = weather.fillna(0)
        weather_map[location] = weather

    return weather_map


def prep_exg_all_bak(data: pd.DataFrame):
    exg_map = defaultdict(lambda: defaultdict(list))
    for lvl1, lvl2, date, val in zip(data['idx_dtl_cd'], data['idx_cd'], data['yymm'], data['ref_val']):
        exg_map[lvl1][lvl2].append((date, val))

    result = pd.DataFrame()
    for key1, val1 in exg_map.items():
        for key2, val2 in val1.items():
            temp = pd.DataFrame(val2, columns=['yymmdd', key2])
            temp = temp.sort_values(by='yymmdd')
            if len(result) == 0:
                result = pd.concat([result, temp], axis=1, join='outer')
            else:
                result = pd.merge(result, temp, on='yymmdd')

    result.columns = [col.lower() for col in result.columns]
    result.loc[:, 'yymmdd'] = result.loc[:, 'yymmdd'].astype(np.int64)
    result = result.fillna(0)

    return result


def make_data_version(data_version: str) -> pd.DataFrame:
    today = date.today()
    monday = today - timedelta(days=today.weekday())  # Convert this week monday
    monday = date.strftime(monday, '%Y%m%d')
    data = {
        'project_cd': ['ENT001'],
        'data_vrsn_cd': [data_version],
        'from_date': [data_version.split('-')[0]],
        'to_date': [data_version.split('-')[1]],
        'exec_date': [monday],
        'use_yn': 'Y',
        'create_user_cd': ['SYSTEM']
    }
    df = pd.DataFrame(data)

    return df


def fill_na(data: pd.DataFrame, chk_list: list) -> pd.DataFrame:
    for col in chk_list:
        if col in list(data.columns):
            data[col] = data[col].fillna('')

    return data


def remove_special_character(data: pd.DataFrame, feature: str):
    feat = deepcopy(data[feature])
    # Chinese characters
    # feat = feat.str.replace('愛', '애')
    # feat = feat.str.replace('入', '인')
    # feat = feat.str.replace('月', '월')

    # Special characters
    feat = feat.str.replace('ℓ', 'l')
    feat = feat.str.replace('㎖', 'ml')
    # feat = feat.str.replace('%', 'PCT')
    # feat = feat.str.replace('&', ' ')
    feat = feat.str.replace('*', ' ')
    feat = feat.str.replace('+', ' ')
    feat = feat.str.replace('-', ' ')
    feat = feat.str.replace('_', ' ')
    feat = feat.str.replace('=', ' ')
    feat = feat.str.replace(',', ' ')
    feat = feat.str.replace('#', ' ')
    feat = feat.str.replace('/', ' ')
    feat = feat.str.replace('\\', ' ')
    feat = feat.str.replace('?', ' ')
    feat = feat.str.replace(':', ' ')
    feat = feat.str.replace('^', ' ')
    feat = feat.str.replace('$', ' ')
    feat = feat.str.replace('.', ' ')
    feat = feat.str.replace('@', ' ')
    feat = feat.str.replace('※', ' ')
    feat = feat.str.replace('~', ' ')
    feat = feat.str.replace('ㆍ', ' ')
    feat = feat.str.replace('!', ' ')
    feat = feat.str.replace('』', ' ')
    feat = feat.str.replace('‘', ' ')
    feat = feat.str.replace('|', ' ')
    feat = feat.str.replace('<', ' ')
    feat = feat.str.replace('>', ' ')

    data[feature] = feat

    return data


def conv_col_lower(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = [col.lower() for col in list(data.columns)]

    return data


def customize_accuracy(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    recalculate accuracy
    """
    acc = data[col]

    # customize rule
    acc = np.where(acc > 1, 2 - acc, acc)
    acc = np.where(acc < 0, 0, acc)    # minus accuracy values convert to zero

    data[col] = acc

    return data


def conv_json_to_dict(path: str) -> dict:
    # Opening JSON file
    with open(path) as json_file:
        data = json.load(json_file)

    return data
