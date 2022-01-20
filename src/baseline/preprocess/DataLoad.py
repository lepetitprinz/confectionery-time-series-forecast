import common.util as util

import pandas as pd


class DataLoad(object):
    def __init__(self, io, sql_conf, date: dict, division: str, data_vrsn_cd: str):
        self.io = io
        self.sql_conf = sql_conf
        self.date = date
        self.division = division
        self.data_vrsn_cd = data_vrsn_cd

    def check_data_version(self) -> None:
        data_vrsn_list = self.io.get_df_from_db(sql=self.sql_conf.sql_data_version())
        if self.data_vrsn_cd not in list(data_vrsn_list['data_vrsn_cd']):
            data_vrsn_db = util.make_data_version(data_version=self.data_vrsn_cd)
            # Insert current data version code
            self.io.insert_to_db(df=data_vrsn_db, tb_name='M4S_I110420')
            # Previous data version usage convert to 'N'
            self.io.update_from_db(sql=self.sql_conf.update_data_version(**{'data_vrsn_cd': self.data_vrsn_cd}))

    def load_sales(self) -> pd.DataFrame:
        sales = None
        if self.division == 'SELL_IN':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_in(**self.date['history']))

        elif self.division == 'SELL_OUT':
            sales = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_out_week(**self.date['history']))

        return sales

    def load_mst(self) -> dict:
        cust_grp = self.io.get_df_from_db(sql=self.sql_conf.sql_cust_grp_info())
        item_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())
        cal_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())
        sales_matrix = self.io.get_df_from_db(sql=self.sql_conf.sql_sales_matrix())

        # Load Algorithm & Hyper-parameter Information
        model_mst = self.io.get_df_from_db(sql=self.sql_conf.sql_algorithm(**{'division': 'FCST'}))
        model_mst = model_mst.set_index(keys='model').to_dict('index')

        param_grid = self.io.get_df_from_db(sql=self.sql_conf.sql_best_hyper_param_grid())
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

    def load_exog(self, info: dict) -> pd.DataFrame:
        # dtype = {
        #     'IDX_CD': object,
        #     'IDX_DTL_CD': object,
        #     'YYMM': object,
        #     'REF_VAL': np.int16
        # }

        exog = self.io.get_df_from_db(sql=self.sql_conf.sql_exg_data(**info))

        return exog

    def filter_new_item(self, sales: pd.DataFrame) -> pd.DataFrame:
        old_item = self.io.get_df_from_db(sql=self.sql_conf.sql_old_item_list())
        old_item = list(old_item.values)

        sales_filtered = sales[sales['sku'].isin(old_item)]

        return sales_filtered
