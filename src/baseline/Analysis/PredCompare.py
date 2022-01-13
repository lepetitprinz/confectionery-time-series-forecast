import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# font_path = r'C:\Windows\Fonts\NanumBarunGothic.ttf'
# font_name = fm.FontProperties(fname=font_path).get_name()
# matplotlib.rc('font', family=font_name)
plt.style.use('fivethirtyeight')


class PredCompare(object):
    col_fixed = ['division_cd', 'start_week_day', 'week']

    def __init__(self, data_cfg: dict):
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.root_path = os.path.join('..', '..', 'analysis')
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

        # Data configuration
        self.data_cfg = data_cfg
        self.division = data_cfg['division']
        self.week_compare = 1
        self.date_sales = {}
        self.data_vrsn_cd = ''
        self.hist_date_range = []
        self.level = {}
        self.hrchy = {}

        # Evaluation configuration
        self.eval_threshold = 0.3    # percent
        self.filter_n_threshold = 5

    def run(self):
        # Initialize information
        self.init()

        # Load the dataset
        sales_compare, sales_hist, pred = self.load_dataset()

        # Resample data
        sales_compare = self.resample_sales(data=sales_compare)
        sales_hist = self.resample_sales(data=sales_hist)

        # Compare result
        result = self.compare_result(sales=sales_compare, pred=pred)

        # Convert
        sales_hist = self.conv_to_datetime(data=sales_hist, col='start_week_day')
        # sales_hist = sales_hist.set_index('start_week_day')
        result = self.conv_to_datetime(data=result, col='start_week_day')

        # Filter n by accuracy
        result = self.filter_n_by_accuracy(data=result)

        # Draw plots
        self.draw_plot(result=result, sales=sales_hist)

    def init(self) -> None:
        self.set_date()    # Set date
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()

    def set_date(self) -> None:
        if self.data_cfg['cycle_yn']:
            self.date_sales = self.calc_sales_date()
            self.data_vrsn_cd = self.date_sales['hist']['from'] + '-' + self.date_sales['hist']['to']
            self.hist_date_range = pd.date_range(
                start=self.date_sales['hist']['from'],
                end=self.date_sales['hist']['to'],
                freq='w'
            )
        else:
            self.date_sales = self.data_cfg['date']
            self.data_vrsn_cd = self.data_cfg['data_vrsn_cd']
            self.hist_date_range = pd.date_range(
                start=self.date_sales['hist']['from'],
                end=self.date_sales['hist']['to'],
                freq='W-MON'
            )

    def set_level(self,  item_lvl: int) -> None:
        level = {
            'cust_lvl': 1,    # Fixed
            'item_lvl': item_lvl,
        }
        self.level = level

    def set_hrchy(self) -> None:
        self.hrchy = {
            'cnt': 0,
            'key': "C" + str(self.level['cust_lvl']) + '-' + "P" + str(self.level['item_lvl']) + '-',
            'lvl': {
                'cust': self.level['cust_lvl'],
                'item': self.level['item_lvl'],
                'total': self.level['cust_lvl'] + self.level['item_lvl']
            },
            'list': {
                'cust': self.common['hrchy_cust'].split(','),
                'item': [config.HRCHY_CD_TO_DB_CD_MAP.get(col, 'item_cd') for col
                         in self.common['hrchy_item'].split(',')[:self.level['item_lvl']]]
            }
        }
        self.hrchy['apply'] = self.hrchy['list']['cust'] + self.hrchy['list']['item']

    def calc_sales_date(self):
        today = datetime.date.today()
        today = today - datetime.timedelta(days=today.weekday())

        # History dates
        hist_from = today - datetime.timedelta(days=int(self.common['week_hist']) * 7 + 7)
        hist_to = today - datetime.timedelta(days=1 + 7)

        hist_from = hist_from.strftime('%Y%m%d')
        hist_to = hist_to.strftime('%Y%m%d')

        # Compare dates
        compare_from = today - datetime.timedelta(days=self.week_compare * 7)
        compare_to = today - datetime.timedelta(days=1)

        compare_from = compare_from.strftime('%Y%m%d')
        compare_to = compare_to.strftime('%Y%m%d')

        date = {
            'hist': {
                'from': hist_from,
                'to': hist_to
            },
            'compare': {
                'from': compare_from,
                'to': compare_to
            }
        }

        return date

    def load_dataset(self):
        # load sales dataset
        info_sales_compare = {
            'division_cd': self.division,
            'start_week_day': self.date_sales['compare']['from']
        }
        sales_compare = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales_compare = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales_compare = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_compare(**info_sales_compare))

        info_sales_hist = {
            'division_cd': self.division,
            'from': self.date_sales['hist']['from'],
            'to': self.date_sales['hist']['to']
        }
        sales_hist = None
        if self.division == 'SELL_IN':  # Sell-In Dataset
            sales_hist = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))
        elif self.division == 'SELL_OUT':  # Sell-Out Dataset
            sales_hist = self.io.get_df_from_db(sql=self.sql_conf.sql_sell_week_hist(**info_sales_hist))

        # load prediction dataset
        info_pred = {
            'data_vrsn_cd': self.data_vrsn_cd,
            'division_cd': self.division,
            'fkey': self.hrchy['key'][:-1],
            'yymmdd': self.date_sales['compare']['from'],
        }
        pred = self.io.get_df_from_db(sql=self.sql_conf.sql_pred_best(**info_pred))
        pred = self.filter_col(data=pred)

        return sales_compare, sales_hist, pred

    def filter_col(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_col = self.hrchy['apply'] + self.col_fixed + ['pred']
        data = data[filter_col]

        return data

    def resample_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        grp_col = self.hrchy['apply'] + self.col_fixed
        data = data.groupby(by=grp_col).sum()
        data = data.reset_index()

        return data

    def fill_missing_date(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        idx_add = list(set(self.hist_date_range) - set(df[col]))
        data_add = np.zeros(len(idx_add))
        # df_add = pd.DataFrame(data_add, index=idx_add, columns=df.columns)
        df_add = pd.DataFrame(np.array([idx_add, data_add]).T, columns=df.columns)
        df = df.append(df_add)
        df = df.sort_values(by=col)

        return df

    def merge_result(self, sales: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        merge_col = self.hrchy['apply'] + self.col_fixed
        merged = pd.merge(
            sales,
            pred,
            on=merge_col,
            how='left'
        )

        return merged

    def compare_result(self, sales: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        # Merge dataset
        data = self.merge_result(sales=sales, pred=pred)

        # Calculate the accuracy
        data = self.calc_accuracy(data=data)
        data = self.eval_result(data=data)

        path = os.path.join(self.root_path, self.data_vrsn_cd + '_' + self.division + '.csv')
        data.to_csv(path, index=False, encoding='cp949')

        return data

    @staticmethod
    def calc_accuracy(data) -> pd.DataFrame:
        conditions = [
            data['sales'] == data['pred'],
            data['sales'] == 0,
            data['sales'] != data['pred']
        ]
        values = [1, 0, data['pred'] / data['sales']]
        accuracy = np.select(conditions, values)
        data['accuracy'] = accuracy

        return data

    def eval_result(self, data) -> pd.DataFrame:
        conditions = [
            data['accuracy'] < 1 - self.eval_threshold,
            data['accuracy'] > 1 + self.eval_threshold
        ]
        values = ['N', 'N']
        success = np.select(conditions, values, default=0)
        success = np.where(success == '0', 'Y', success)
        data['success'] = success

        print(f"Prediction Success: {len(data[data['success'] == 'Y'])}")
        print(f"Prediction Fail: {len(data[data['success'] == 'N'])}")

        return data

    def filter_n_by_accuracy(self, data) -> dict:
        data['rank_score'] = np.abs(data['accuracy']-1)
        data = data.sort_values(by=['rank_score'])
        data = data.drop(columns=['rank_score'])

        data = data[data['accuracy'] != 1]
        best = data.iloc[:self.filter_n_threshold, :]
        worst = data.iloc[-self.filter_n_threshold:, :]
        worst = worst.fillna(0)

        # zero = data[data['accuracy'] == 0]
        # zero = zero.iloc[:self.filter_n_threshold, :]
        # zero = zero.fillna(0)

        # result = {'best': best, 'worst': worst, 'zero': zero}
        result = {'best': best, 'worst': worst}

        return result

    def draw_plot(self, result: dict, sales: pd.DataFrame):
        item_lvl_col = self.hrchy['list']['item'][-1]
        for kind, data in result.items():
            for cust_grp, item_lvl_cd, date, pred, sale in zip(
                    data['cust_grp_cd'], data[item_lvl_col], data['start_week_day'], data['pred'], data['sales']):
                filtered = sales[(sales['cust_grp_cd'] == cust_grp) & (sales[item_lvl_col] == item_lvl_cd)]
                filtered = filtered[['start_week_day', 'sales']]
                filtered = self.fill_missing_date(df=filtered, col='start_week_day')

                # fix, ax1 = plt.subplots(1, 1, figsize=(15, 6))
                fix, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                # ax1.plot(filtered['start_week_day'], filtered['sales'],
                #          linewidth=1.3, color='grey', label='history')

                if len(filtered) > 0:
                    df_sales = self.connect_dot(hist=filtered, date=date, value=sale)
                    df_pred = self.connect_dot(hist=filtered, date=date, value=pred)
                    ax1.plot(df_sales['start_week_day'], df_sales['qty'], color='darkblue',
                             linewidth=1.3, alpha=0.7)
                    ax1.plot(df_pred['start_week_day'], df_pred['qty'], color='firebrick',
                             linewidth=2, alpha=0.8, linestyle='dotted')

                    # Ticks
                    # plt.xlim(xmin=df_sales['start_week_day'].tolist()[0],
                    #          xmax=df_sales['start_week_day'].tolist()[-1])
                    plt.xticks(rotation=30)
                    plt.ylim(ymin=-5)
                    date_str = datetime.datetime.strftime(date, '%Y%m%d')
                    plt.title(f'{date_str}: Cust Group: {cust_grp} / Brand Code: {item_lvl_cd}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        self.root_path, 'plot', date_str, date_str + '_' + self.division + '_' + kind + '_' + \
                        str(cust_grp) + '_' + str(item_lvl_cd) + '.png'))

                    plt.close()

    @staticmethod
    def connect_dot(hist: pd.DataFrame, date, value):
        hist = hist[['start_week_day', 'sales']]
        hist = hist.rename(columns={'sales': 'qty'})
        hist = hist.sort_values(by='start_week_day')

        result = pd.DataFrame({'start_week_day': [date], 'qty': [value]})
        result = result.append(hist.iloc[-1, :], ignore_index=True)

        return result

    @staticmethod
    def conv_to_datetime(data: pd.DataFrame, col: str) -> pd.DataFrame:
        data[col] = pd.to_datetime(data[col])

        return data
