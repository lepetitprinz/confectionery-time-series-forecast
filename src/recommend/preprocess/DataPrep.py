import pandas as pd


class DataPrep(object):
    def __init__(self, item_col: str, meta_col: str):
        self.item_col = item_col
        self.meta_col = meta_col
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

    @staticmethod
    def conv_to_db_table(data: list):
        db = []
        for item, ranks in data:
            for i, rank in enumerate(ranks):
                db.append([item, i+1, rank[0], rank[1]])

        db = pd.DataFrame(db, columns=['item_cd', 'rank', 'sim_item_cd', 'score'])
        db['project_cd'] = 'ENT001'
        db['create_user_cd'] = 'SYSTEM'

        return db
