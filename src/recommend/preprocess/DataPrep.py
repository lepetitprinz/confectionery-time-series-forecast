import pandas as pd


class DataPrep(object):
    def __init__(self, common: dict, data_vrsn_cd: str, data_vrsn_list: pd.DataFrame, cal: pd.DataFrame):
        self.common = common
        self.data_vrsn_cd = data_vrsn_cd
        self.data_vrsn_list = data_vrsn_list
        self.cal = cal
        self.col_str = ['sku_cd', 'bom_cd']

    def preprocess(self, data: pd.DataFrame):
        data = self.conv_type(data=data)

        return data

    def conv_type(self, data: pd.DataFrame):
        # Convert to string type
        for col in self.col_str:
            data[col] = data[col].astype(str)

        return data

    def check_na(self):
        pass

    def make_db_format(self, data: list):
        similar = []
        for item, ranks in data:
            for i, rank in enumerate(ranks):
                similar.append([item, i+1, rank[0], rank[1]])

        result = pd.DataFrame(similar, columns=['item_cd', 'rank', 'sim_item_cd', 'score'])
        result['project_cd'] = self.common['project_cd']
        result['data_vrsn_cd'] = self.data_vrsn_cd
        result['create_user_cd'] = 'SYSTEM'

        # result = pd.merge(result, self.cal, on)

        return result
