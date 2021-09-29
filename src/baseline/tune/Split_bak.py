import common.config as config

import os
import numpy as np
import pandas as pd


class Split_bak(object):
    def __init__(self, data_vrsn_cd: str, division_cd: str, lvl: dict):
        self.fix_col = ['data_vrsn_cd', 'division_cd', 'stat_cd', 'week', 'yymmdd']
        self.target_col = 'qty'
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division_cd

        # Data Level Configuration
        # Ratio
        self.lvl = lvl
        self.ratio_hrchy = config.LVL_CD_LIST[:lvl['lvl_ratio']]
        self.ratio_lvl = lvl['lvl_ratio']
        self.ratio_cd = config.LVL_MAP[lvl['lvl_ratio']]
        self.ratio = None

        # Split
        self.split_hrchy = config.LVL_CD_LIST[:lvl['lvl_split']]

    def filter_col(self, df: pd.DataFrame, division: str):
        cols = None
        if division == 'ratio':
            cols = self.fix_col + self.ratio_hrchy + [self.target_col]
        elif division == 'split':
            cols = self.fix_col + self.split_hrchy + [self.target_col]
        df = df[cols]

        return df

    def run(self, df_ratio: pd.DataFrame, df_split: pd.DataFrame):
        df_ratio = self.filter_col(df=df_ratio, division='ratio')
        df_split = self.filter_col(df=df_split, division='split')
        df_ratio = self.calc_ratio(hrchy=self.ratio_hrchy, hrchy_lvl=self.ratio_lvl - 1, data=df_ratio)
        df_ratio_list = self.hrchy_recursion_with_key(hrchy_lvl=self.ratio_lvl-1, fn=self.drop_qty, df=df_ratio)
        df_ratio = self.concat_df(df_list=df_ratio_list)

        merged = pd.merge(df_split, df_ratio, how='right',
                          on=self.fix_col + self.split_hrchy)
        merged = merged.sort_values(by=self.fix_col + self.split_hrchy)
        merged['split_qty'] = np.round(merged[self.target_col] * merged['rate'])

        title = config.LVL_CD_LIST[self.lvl['lvl_ratio']-1] + '_to_' + config.LVL_CD_LIST[self.lvl['lvl_split']-1]
        merged.to_csv(os.path.join('..', '..', 'tune', title + '.csv'), index=False)

    @staticmethod
    def concat_df(df_list: list):
        concat = pd.DataFrame()
        for df in df_list:
            concat = pd.concat([concat, df], axis=0)

        return concat

    def drop_qty(self, hrchy: list, data: pd.DataFrame):
        data[self.ratio_cd] = hrchy[-1]
        data = data.drop(columns=[self.target_col, 'sum'])

        return data

    def calc_ratio(self, hrchy, hrchy_lvl, data, cd=None, lvl=0):
        rate = {}
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
                result = self.calc_ratio(hrchy=hrchy, hrchy_lvl=hrchy_lvl, data={code: sliced},
                                         cd=code, lvl=lvl + 1)
                rate[code] = result

        elif lvl == hrchy_lvl:
            # rating list
            temp = []
            for code in code_list:
                sliced = None
                if isinstance(data, pd.DataFrame):
                    sliced = data[data[col] == code]
                    sliced = sliced.reset_index(drop=True)
                elif isinstance(data, dict):
                    sliced = data[cd][data[cd][col] == code]
                    sliced = sliced.reset_index(drop=True)
                # sliced = sliced.sort_values(by=self.fix_col + self.split_hrchy)
                    sliced = sliced.rename(columns={self.target_col: self.target_col + '_' + code})
                temp.append((code, sliced))

            if len(temp) > 1:
                merged = temp[0][1]
                for _, df in temp[1:]:
                    merged = pd.merge(merged, df, on=self.fix_col, suffixes=('', '_DROP'))

                # filter qty columns
                qty_cols = [col for col in merged.columns if 'qty' in col]
                qty = merged[self.fix_col + self.split_hrchy + qty_cols]
                qty.loc[:, 'sum'] = qty.sum(axis=1)

                result = {}
                for code, df in temp:
                    qty_col = self.target_col + '_' + code
                    sliced = qty[self.fix_col + self.split_hrchy + [qty_col] + ['sum']]
                    # sliced['rate'] = sliced[qty_col] / sliced['sum']
                    sliced.loc[:, 'rate'] = sliced[qty_col] / sliced['sum']
                    sliced = sliced.rename(columns={qty_col: self.target_col})
                    result[code] = sliced

            else:
                code, df = temp[0]
                df = df.rename(columns={self.target_col + '_' + code: self.target_col})
                df['rate'] = 1
                df['sum'] = df[self.target_col]
                result = {code: df}

            return result

        return rate

    def hrchy_recursion_with_key(self, hrchy_lvl, fn=None, df=None, val=None, lvl=0, hrchy=[]):
        if lvl == 0:
            temp = []
            for key, val in df.items():
                hrchy.append(key)
                result = self.hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val,
                                                       lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key)

        elif lvl < hrchy_lvl:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = self.hrchy_recursion_with_key(hrchy_lvl=hrchy_lvl, fn=fn, val=val_hrchy,
                                                       lvl=lvl+1, hrchy=hrchy)
                temp.extend(result)
                hrchy.remove(key_hrchy)

            return temp

        elif lvl == hrchy_lvl:
            temp = []
            for key_hrchy, val_hrchy in val.items():
                hrchy.append(key_hrchy)
                result = fn(hrchy, val_hrchy)
                temp.append(result)
                hrchy.remove(key_hrchy)

            return temp

        return temp
