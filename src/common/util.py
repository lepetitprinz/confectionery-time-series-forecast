import os
import numpy as np
import pandas as pd
from collections import defaultdict


def group(data, hrchy, hrchy_lvl,  cd=None, lvl=0) -> dict:
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
            result = group(hrchy=hrchy, hrchy_lvl=hrchy_lvl, data={code: sliced},
                           cd=code, lvl=lvl + 1)
            grp[code] = result

    elif lvl == hrchy_lvl:
        temp = {}
        for code in code_list:
            sliced = None
            if isinstance(data, dict):
                sliced = data[cd][data[cd][col] == code]
            if isinstance(data, pd.DataFrame):
                sliced = data[data[col] == code]
            temp[code] = sliced

        return temp

    return grp


def hrchy_recursion(hrchy_lvl, fn=None, df=None, val=None, lvl=0) -> dict:
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


def hrchy_recursion_with_key(hrchy_lvl, fn=None, df=None, val=None, lvl=0) -> None:
    if lvl == 0:
        for key, val in df.items():
            hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val, lvl=lvl + 1)

    elif lvl < hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy, lvl=lvl + 1)

    elif lvl == hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            fn(key_hrchy, val_hrchy)


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


def make_path_baseline(module: str, division: str, data_vrsn: str, hrchy_lvl: str, step: str, extension: str):
    path = os.path.join('..', '..', module, division + '_' + data_vrsn + '_' + str(hrchy_lvl) + step + '.' + extension)

    return path


def make_path_sim(module: str, division: str, step: str, extension: str):
    path = os.path.join('..', '..', module, division + '_' + step + '.' + extension)

    return path


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


def prep_exg_partial(data: pd.DataFrame) -> pd.DataFrame:
    item_cust = data['idx_dtl_cd'].str.split('_', 1, expand=True)
    item_cust.columns = ['sku_cd', 'cust_grp_cd']

    exg_partial = pd.concat([data, item_cust], axis=1)
    exg_partial = exg_partial.drop(columns=['idx_cd', 'idx_dtl_cd'])

    return exg_partial
