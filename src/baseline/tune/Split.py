import common.config as config

import pandas as pd


class Split(object):
    def __init__(self, data_vrsn_cd: str, division_cd: str, lvl: dict):
        self.fix_col = ['data_vrsn_cd', 'division_cd', 'stat_cd', 'week', 'yymmdd']
        self.target_col = 'qty'
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division_cd

        # Data Level Configuration
        # Ratio
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

        print("")
        return merged

    @staticmethod
    def concat_df(df_list: list):
        concat = pd.DataFrame()
        for df in df_list:
            concat = pd.concat([concat, df], axis=0)

        return concat

    def drop_qty(self, hrchy: list, data: pd.DataFrame):
        data = data.drop(columns=[self.target_col])
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
                temp.append((code, sliced))

            if len(temp) > 1:
                concat = pd.DataFrame()
                for df in temp:
                    other = df[1].sort_values(by=self.fix_col + self.split_hrchy)
                    concat = pd.concat([concat, other], axis=1)

                # filter qty columns
                qty_cols = [col for col in concat.columns if 'qty' in col]
                qty_sum = concat[qty_cols]
                qty_sum['sum'] = qty_sum.sum(axis=1)

                for t in temp:
                    t[1]['rate'] = t[1]['qty'] / qty_sum['sum']

                result = {}
                for rated in temp:
                    result[rated[0]] = rated[1]

            else:
                temp[0][1]['rate'] = 1
                result = {temp[0][0]: temp[0][1]}

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