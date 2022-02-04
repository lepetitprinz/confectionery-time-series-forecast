import pandas as pd


class DataPrep(object):
    def __init__(self, common: dict, data_vrsn_cd: str, data_vrsn_list: pd.DataFrame,
                 item_mst: pd.DataFrame, cal: pd.DataFrame):
        self.cal = cal    # Calendar dataset
        self.common = common    # Common information
        self.col_str = ['sku_cd', 'bom_cd']
        self.item_mst = item_mst    # Item master
        self.data_vrsn_cd = data_vrsn_cd    # Data version code
        self.data_version_info = data_vrsn_list[data_vrsn_list['data_vrsn_cd'] == data_vrsn_cd]

    def preprocess(self, data: pd.DataFrame):
        # Change data types
        data = self.conv_type(data=data)

        return data

    def conv_type(self, data: pd.DataFrame):
        # Convert columns to string type
        for col in self.col_str:
            data[col] = data[col].astype(str)

        return data

    # Re-rank the similarity
    def filter_rerank_by_line(self, data: list, filter_n: int):
        data = self.conv_to_df(data=data)           # Convert list to dataframe
        data = self.add_item_mst_info(data=data)    # Add item master
        data = self.filter_line(data=data)          # Filter line code
        data = self.re_rank(data=data, filter_n=filter_n)    # Re-rank by each line

        return data

    @staticmethod
    def conv_to_df(data: list) -> pd.DataFrame:
        similar = []
        for item, ranks in data:
            for i, rank in enumerate(ranks):
                similar.append([item, i + 1, rank[0], rank[1]])

        df = pd.DataFrame(similar, columns=['item_cd', 'rank', 'sim_item_cd', 'score'])

        return df

    # Add item master information
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
    # Filter Line list
    def filter_line(data: pd.DataFrame) -> pd.DataFrame:
        data = data[data['item_line_cd'] == data['sim_item_line_cd']]
        data = data.drop(columns=['item_line_cd', 'sim_item_line_cd'])    # Drop unnecessary columns
        data = data.sort_values(by=['item_cd', 'rank'])    # Sort rank values
        data = data.reset_index(drop=True)    # Reset index

        return data

    @staticmethod
    def re_rank(data: pd.DataFrame, filter_n: int) -> pd.DataFrame:
        data['re_rank'] = data.groupby(by='item_cd')['rank'].rank(method='min')
        data['re_rank'] = data['re_rank'].astype(int)    # Change data type(str -> int)
        data = data.drop(columns=['rank'])    # Drop unnecessary columns
        data = data.rename(columns={'re_rank': 'rank'})
        data = data[data['rank'] <= filter_n]    # Filter top n
        data = data.reset_index(drop=True)    # Reset index

        return data

    # Add information on the result
    def add_db_info(self, data: pd.DataFrame):
        data['project_cd'] = self.common['project_cd']    # Add project code
        data['data_vrsn_cd'] = self.data_vrsn_cd    # Add data version
        data['yymmdd'] = self.data_version_info['exec_date'].values[0]    #
        data['yy'] = data['yymmdd'].str.slice(stop=4)
        data['create_user_cd'] = 'SYSTEM'    # User code

        # Merge calendar information
        result = pd.merge(data, self.cal, how='left', on='yymmdd')

        return result
