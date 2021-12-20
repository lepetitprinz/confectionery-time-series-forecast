import pandas as pd


class DataPrep(object):
    def __init__(self, common: dict, data_vrsn_cd: str, data_vrsn_list: pd.DataFrame,
                 item_mst: pd.DataFrame, cal: pd.DataFrame):
        self.common = common
        self.data_vrsn_cd = data_vrsn_cd
        self.data_version_info = data_vrsn_list[data_vrsn_list['data_vrsn_cd'] == data_vrsn_cd]
        self.item_mst = item_mst
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

    def filter_rerank_by_line(self, data: list, filter_n: int):
        data = self.conv_to_df(data=data)
        data = self.add_item_mst_info(data=data)
        data = self.filter_line(data=data)
        data = self.re_rank(data=data, filter_n=filter_n)

        return data

    @staticmethod
    def conv_to_df(data: list) -> pd.DataFrame:
        similar = []
        for item, ranks in data:
            for i, rank in enumerate(ranks):
                similar.append([item, i + 1, rank[0], rank[1]])

        df = pd.DataFrame(similar, columns=['item_cd', 'rank', 'sim_item_cd', 'score'])

        return df

    def add_item_mst_info(self, data: pd.DataFrame):
        item_mst = self.item_mst[['sku_cd', 'line_cd']]
        item_mst = item_mst.rename(columns={'line_cd': 'item_line_cd', 'sku_cd': 'item_cd'})
        sim_item_mst = self.item_mst[['sku_cd', 'line_cd']]
        sim_item_mst = sim_item_mst.rename(columns={'line_cd': 'sim_item_line_cd', 'sku_cd': 'sim_item_cd'})

        # Merge item information
        merged = pd.merge(data, item_mst, how='inner', on='item_cd')
        merged = pd.merge(merged, sim_item_mst, how='inner', on='sim_item_cd')

        return merged

    @staticmethod
    def filter_line(data: pd.DataFrame):
        data = data[data['item_line_cd'] == data['sim_item_line_cd']]
        data = data.drop(columns=['item_line_cd', 'sim_item_line_cd'])
        data = data.sort_values(by=['item_cd', 'rank'])
        data = data.reset_index(drop=True)

        return data

    @staticmethod
    def re_rank(data: pd.DataFrame, filter_n: int):
        data['re_rank'] = data.groupby(by='item_cd')['rank'].rank(method='min')
        data['re_rank'] = data['re_rank'].astype(int)
        data = data.drop(columns=['rank'])
        data = data.rename(columns={'re_rank': 'rank'})
        data = data[data['rank'] <= filter_n]
        data = data.reset_index(drop=True)

        return data

    def add_db_info(self, data: pd.DataFrame):
        data['project_cd'] = self.common['project_cd']
        data['data_vrsn_cd'] = self.data_vrsn_cd
        data['create_user_cd'] = 'SYSTEM'
        data['yymmdd'] = self.data_version_info['exec_date'].values[0]
        data['yy'] = data['yymmdd'].str.slice(stop=4)

        result = pd.merge(data, self.cal, how='left', on='yymmdd')

        return result
