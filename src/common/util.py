import os
import pandas as pd
from collections import defaultdict

def group(hrchy, hrchy_lvl, data, cd=None, lvl=0) -> dict:
    grp = {}
    col = hrchy[lvl][1]

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
            if isinstance(data, pd.DataFrame):
                sliced = data[data[col] == code]
            elif isinstance(data, dict):
                sliced = data[cd][data[cd][col] == code]
            temp[code] = sliced

        return temp

    return grp


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
            if len(val_hrchy) > 2:
                result = fn(val_hrchy)
                temp[key_hrchy] = result
        return temp

    return temp


def hrchy_recursion_with_key(hrchy_lvl, fn=None, df=None, val=None, lvl=0, hrchy=[]):
    if lvl == 0:
        temp = []
        for key, val in df.items():
            hrchy.append(key)
            result = hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val,
                                              lvl=lvl+1, hrchy=hrchy)
            temp.extend(result)
            hrchy.remove(key)

    elif lvl < hrchy_lvl:
        temp = []
        for key_hrchy, val_hrchy in val.items():
            hrchy.append(key_hrchy)
            result = hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy,
                                              lvl=lvl+1, hrchy=hrchy)
            temp.extend(result)
            hrchy.remove(key_hrchy)

        return temp

    elif lvl == hrchy_lvl:
        for key_hrchy, val_hrchy in val.items():
            if len(val_hrchy) > 2:
                hrchy.append(key_hrchy)
                temp = fn(hrchy, val_hrchy)
                hrchy.remove(key_hrchy)
        return temp

    return temp


def make_path(module: str, division: str, hrchy_lvl: int, step: str, data_type: str):
    path = os.path.join('..', module, division + '_' + str(hrchy_lvl) + '_' + step + '.' + data_type)

    return path


def make_lvl_key_val_map(df: pd.DataFrame, lvl: str, key: str, val: str):
    result = defaultdict(lambda: defaultdict(dict))
    for lvl, key, val in zip(df[lvl], df[key], df[val]):
        result[lvl][key] = val

    return result

