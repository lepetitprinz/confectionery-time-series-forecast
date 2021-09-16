import common.config as config

import pandas as pd


class Split(object):
    def __init__(self, data_vrsn_cd: str, division_cd: str, lvl_ratio: int):
        self.fix_col = ['data_vrsn_cd', 'division_cd', 'stat_cd', 'week', 'yymmdd']
        self.target_col = 'qty'
        self.data_vrsn_cd = data_vrsn_cd
        self.division = division_cd

        # Data Level Configuration
        self.lvl_ratio = lvl_ratio
        self.rating_cd = config.LVL_MAP[lvl_ratio]
        self.hrchy = config.LVL_CD_LIST[:lvl_ratio]

    def filter_col(self, df: pd.DataFrame):
        cols = self.fix_col + self.hrchy + [self.target_col]
        df = df[cols]

        return df

    def run(self, df_ratio: pd.DataFrame, df_split: pd.DataFrame):
        df_ratio = self.filter_col(df=df_ratio)
        ratio = self.calc_ratio(hrchy=self.hrchy, hrchy_lvl=self.lvl_ratio - 1, data=df_ratio)

    def split_ratio(self, hrchy, hrchy_lvl, data, cd=None, lvl=0):
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
                result = self.split_ratio(hrchy=hrchy, hrchy_lvl=hrchy_lvl, data={code: sliced},
                                          cd=code, lvl=lvl + 1)
                rate[code] = result

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
                temp.append((code, sliced))

            if len(temp) > 1:
                merged = None
                first = temp[0]
                for other in temp[1:]:
                    merged = pd.merge(first[1], other[1], how='inner', on=self.fix_col + self.hrchy[:-1])

                # filter qty columns
                qty_cols = [col for col in merged.columns if 'qty' in col]
                qty_sum = merged[qty_cols]
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