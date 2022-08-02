import common.util as util

import pandas as pd
from typing import Union, Dict, List, Tuple, Any


class DataLoad(object):
    def __init__(
            self,
            io,
            sql_conf,
            date: dict,
            division: str,
            data_vrsn_cd: str
    ):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        :param date: Date information
        :param division: Division (SELL-IN/SELL-OUT)
        :param data_vrsn_cd: Data version
        """
        self.io = io
        self.sql_conf = sql_conf
        self.date = date
        self.division = division
        self.data_vrsn_cd = data_vrsn_cd

    # Check that data version exist
    def check_data_version(self) -> None:
        # Load all of data version list from db
        data_vrsn_list = self.io.get_df_from_db(sql=self.sql_conf.sql_data_version())
        if self.data_vrsn_cd not in list(data_vrsn_list['data_vrsn_cd']):
            # Make new data version
            data_vrsn_db = util.make_data_version(data_version=self.data_vrsn_cd)
            # Insert current data version code
            self.io.insert_to_db(df=data_vrsn_db, tb_name='M4S_I110420')
            # Previous data version usage convert to 'N'
            self.io.update_from_db(sql=self.sql_conf.update_data_version(**{'data_vrsn_cd': self.data_vrsn_cd}))

    # Load sales history dataset
    def load_sales(self) -> pd.DataFrame:
        sales = None
        if self.division == 'SELL_IN':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date['history']))

        elif self.division == 'SELL_OUT':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**self.date['history']))

        # Filter New items
        sales = self.filter_new_item(sales=sales)

        return sales

    # Load all of master dataset
    def load_mst(self) -> Dict[str, Any]:
        cust_grp = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_grp_info())    # SP1 master
        item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())    # Item master
        cal_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())    # Calendar master
        sales_matrix = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_matrix())    # Sales matrix master

        # Load Algorithm & Hyper-parameter Information
        # Algorithm master
        model_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_algorithm(**{'division': 'FCST'}))
        model_mst = model_mst.set_index(keys='model').to_dict('index')    # Convert to dictionary data type

        # Hyper parameter master
        param_grid = self.io.get_df_from_db(sql=self.sql_conf.sql_best_hyper_param_grid())    # Load information from DB
        # Convert uppercase into lowercase
        param_grid['stat_cd'] = param_grid['stat_cd'].apply(lambda x: x.lower())
        param_grid['option_cd'] = param_grid['option_cd'].apply(lambda x: x.lower())
        param_grid = util.make_lvl_key_val_map(df=param_grid, lvl='stat_cd', key='option_cd', val='option_val')

        mst_info = {
            'cust_grp': cust_grp,
            'item_mst': item_mst,
            'cal_mst': cal_mst,
            'sales_matrix': sales_matrix,
            'model_mst': model_mst,
            'param_grid': param_grid
        }

        return mst_info

    # Load exogenous dataset: weather
    # Exogenous information
    def load_exog(self) -> pd.DataFrame:
        info = {
            'partial_yn': 'N',
            'from': self.date['history']['from'],
            'to': self.date['history']['to']
        }
        exog = self.io.get_df_from_db(sql=self.sql_conf.sql_exg_data(**info))

        return exog

    # Load sell-in distribution store chain
    def load_sales_dist(self) -> pd.DataFrame:
        info = {
            'from': self.date['history']['from'],
            'to': self.date['history']['to']
        }

        sales_dist = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in_dist(**info))

        return sales_dist

    # Filter new item for predict sales on only old item
    def filter_new_item(self, sales: pd.DataFrame) -> pd.DataFrame:
        old_item = self.io.get_df_from_db(sql=self.sql_conf.sql_old_item_list())    # Load old item from DB
        old_item = old_item['item_cd'].tolist()

        sales_filtered = sales[sales['sku_cd'].isin(old_item)]    # Filter new items

        return sales_filtered
