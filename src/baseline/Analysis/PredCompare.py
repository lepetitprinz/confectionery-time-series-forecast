import common.config as config
from common.SqlConfig import SqlConfig
from dao.DataIO import DataIO

import os
from typing import Dict, Tuple
import datetime
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# font_path = r'C:\Windows\Fonts\malgun.ttf'
# font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family='Malgun Gothic')
plt.style.use('fivethirtyeight')


class PredCompare(object):
    col_fixed = ['division_cd', 'start_week_day', 'week']
    # SP1: 도봉 / 안양 / 동울산 / 논산 / 동작 / 진주 / 이마트 / 롯데슈퍼 / 7-11
    pick_sp1 = ['1005', '1022', '1051', '1063', '1107', '1128', '1065', '1073', '1173']

    def __init__(self, exec_cfg: dict, opt_cfg: dict, date_cfg: dict, data_cfg: dict):
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.root_path = data_cfg['root_path']
        self.common = self.io.get_dict_from_db(
            sql=SqlConfig.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

        # Data configuration
        self.exec_cfg = exec_cfg
        self.opt_cfg = opt_cfg
        self.date_cfg = date_cfg
        self.data_cfg = data_cfg
        self.division = data_cfg['division']
        self.date_sales = {}
        self.data_vrsn_cd = ''
        self.hist_date_range = []
        self.level = {}
        self.hrchy = {}
        self.item_code_map = {}

        # Evaluation configuration
        self.week_compare = 1             # Compare week range
        self.sales_threshold = 5          # Minimum sales quantity
        self.eval_threshold = 0.7         # Accuracy: Success or not
        self.filter_top_n_threshold = 4   # Filter top N
        self.filter_acc_rate = 0.1        # Filter accuracy rate

    def run(self) -> None:
        # Initialize information
        self.init()

        # Load the dataset
        sales_compare, sales_hist, pred = self.load_dataset()

        # Preprocess the dataset
        if self.exec_cfg['cls_prep']:
            sales_hist, sales_compare = self.preprocessing(sales_hist=sales_hist, sales_compare=sales_compare)

        # Compare result
        result = None
        if self.exec_cfg['cls_comp']:
            result = self.compare_result(sales=sales_compare, pred=pred)

            if self.opt_cfg['filter_specific_acc_yn']:
                self.filter_specific_accuracy(data=result)

        # Convert
        sales_hist = self.conv_to_datetime(data=sales_hist, col='start_week_day')
        result = self.conv_to_datetime(data=result, col='start_week_day')

        if self.exec_cfg['cls_top_n']:
            if self.opt_cfg['pick_specific_sp1_yn']:
                result_pick = self.pick_specific_sp1(data=result)
                for sp1, data in result_pick.items():
                    # Filter n by accuracy
                    data_filter_n = self.filter_n_by_accuracy(data=data)
                    # Draw plots
                    if self.exec_cfg['cls_graph']:
                        self.draw_plot(result=data_filter_n, sales=sales_hist)

            else:
                # Filter n by accuracy
                result = self.filter_n_by_accuracy(data=result)

                # Draw plots
                if self.exec_cfg['cls_graph']:
                    self.draw_plot(result=result, sales=sales_hist)

    def filter_specific_accuracy(self, data: pd.DataFrame):
        # Filter result by accuracy
        filtered = data[data['accuracy'] < self.filter_acc_rate]

        # Save the result
        path = os.path.join(self.root_path, self.data_vrsn_cd, 'filter', self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '_' + str(self.filter_acc_rate) + '.csv')
        filtered.to_csv(path, index=False, encoding='cp949')

    def preprocessing(self, sales_hist: pd.DataFrame, sales_compare: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Resample data
        sales_compare = self.resample_sales(data=sales_compare)
        sales_hist = self.resample_sales(data=sales_hist)

        # remove zero quantity
        if self.opt_cfg['rm_zero_yn']:
            print("Remove zero sales quantities")
            sales_compare = self.filter_zeros(data=sales_compare, col='sales')

        if self.opt_cfg['filter_sales_threshold_yn']:
            print(f"Filter sales under {self.sales_threshold} quantity")
            sales_compare = self.filter_sales_threshold(hist=sales_hist, recent=sales_compare)

        return sales_hist, sales_compare

    def pick_specific_sp1(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        data_pick = data[data['cust_grp_cd'].isin(self.pick_sp1)]

        # sorting
        data_pick = data_pick.sort_values(by=['cust_grp_cd', 'accuracy'])

        # split by sp1
        splited = {}
        for sp1 in self.pick_sp1:
            temp = data_pick[data_pick['cust_grp_cd'] == sp1]
            if len(temp) > 0:
                splited[sp1] = temp.reset_index(drop=True)

        return splited

    def filter_sales_threshold(self, hist: pd.DataFrame, recent: pd.DataFrame):
        hist_avg = self.calc_avg_sales(data=hist)
        hist_data_level = self.filter_avg_sales_by_threshold(data=hist_avg)
        recent_filtered = self.merged_filtered_data_level(data=recent, data_level=hist_data_level)

        return recent_filtered

    def calc_avg_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        avg = data.groupby(by=self.hrchy['apply']).mean()
        avg = avg.reset_index()

        return avg

    def filter_avg_sales_by_threshold(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data['sales'] >= self.sales_threshold]
        data = data[self.hrchy['apply']]    # Data level columns

        return data

    def merged_filtered_data_level(self, data: pd.DataFrame, data_level: pd.DataFrame):
        merged = pd.merge(data, data_level, on=self.hrchy['apply'], how='inner')

        return merged

    def init(self) -> None:
        self.set_date()    # Set date
        self.set_level(item_lvl=self.data_cfg['item_lvl'])
        self.set_hrchy()
        self.get_item_info()
        self.make_dir()

    def get_item_info(self):
        item_info = self.io.get_df_from_db(sql=self.sql_conf.sql_item_view())
        item_info.columns = [config.HRCHY_CD_TO_DB_CD_MAP.get(col, col) for col in item_info.columns]
        item_info.columns = [config.HRCHY_SKU_TO_DB_SKU_MAP.get(col, col) for col in item_info.columns]
        item_lvl_cd = self.hrchy['apply'][-1]
        item_lvl_nm = item_lvl_cd[:-2] + 'nm'
        item_info = item_info[[item_lvl_cd, item_lvl_nm]].drop_duplicates()
        item_code_map = item_info.set_index(item_lvl_cd)[item_lvl_nm].to_dict()

        self.item_code_map = item_code_map

    def set_date(self) -> None:
        if self.date_cfg['cycle_yn']:
            self.date_sales = self.calc_sales_date()
            self.data_vrsn_cd = self.date_sales['hist']['from'] + '-' + self.date_sales['hist']['to']
            self.hist_date_range = pd.date_range(
                start=self.date_sales['hist']['from'],
                end=self.date_sales['hist']['to'],
                freq='w'
            )
        else:
            self.date_sales = self.date_cfg['date']
            self.data_vrsn_cd = self.date_cfg['data_vrsn_cd']
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

    def make_dir(self) -> None:
        path = os.path.join(self.root_path, self.data_vrsn_cd)
        if not os.path.isdir(path):
            os.mkdir(path=path)
            os.mkdir(path=os.path.join(path, 'result'))
            os.mkdir(path=os.path.join(path, 'plot'))
            os.mkdir(path=os.path.join(path, 'filter'))

    def calc_sales_date(self) -> Dict[str, Dict[str, str]]:
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

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    @staticmethod
    def filter_zeros(data: pd.DataFrame, col: str) -> pd.DataFrame:
        return data[data[col] != 0]

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
            pred,
            sales,
            on=merge_col,
            how='inner'    # Todo: choose left or inner
        )
        merged['pred'] = merged['pred'].fillna(0)
        merged['sales'] = merged['sales'].fillna(0)

        return merged

    def compare_result(self, sales: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        # Merge dataset
        data = self.merge_result(sales=sales, pred=pred)

        # Calculate the accuracy
        data = self.calc_accuracy(data=data)
        data = self.eval_result(data=data)

        item_lvl_cd = self.hrchy['apply'][-1]
        item_lvl_nm = item_lvl_cd[:-2] + 'nm'
        data[item_lvl_nm] = [self.item_code_map.get(code, code) for code in data[item_lvl_cd].values]
        path = os.path.join(self.root_path, self.data_vrsn_cd, 'result', self.data_vrsn_cd + '_' + self.division +
                            '_' + str(self.hrchy['lvl']['item']) + '.csv')
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

        # Additional calculation
        accuracy = np.where(accuracy > 1, 2 - accuracy, accuracy)
        accuracy = np.where(accuracy < 0, 0, accuracy)

        data['accuracy'] = accuracy

        return data

    def eval_result(self, data) -> pd.DataFrame:
        data['success'] = np.where(data['accuracy'] < self.eval_threshold, 'N', 'Y')

        len_total = len(data)
        len_success = len(data[data['success'] == 'Y'])
        len_fail = len(data[data['success'] == 'N'])

        print("---------------------------------")
        print(f"Prediction Total: {len_total}")
        print(f"Prediction Success: {len_success}, {round(len_success/len_total, 3)}")
        print(f"Prediction Fail: {len_fail}, {round(len_fail/len_total, 3)}")
        print("---------------------------------")

        return data

    def filter_n_by_accuracy(self, data) -> Dict[str, pd.DataFrame]:
        data = data.sort_values(by=['accuracy'], ascending=False)
        data = data[data['accuracy'] != 1]    # remove zero equal zero case
        best = data.iloc[:self.filter_top_n_threshold, :]
        worst = data.iloc[-self.filter_top_n_threshold:, :]
        worst = worst.fillna(0)

        result = {'best': best, 'worst': worst}

        return result

    def draw_plot(self, result: dict, sales: pd.DataFrame):
        item_lvl_col = self.hrchy['list']['item'][-1]
        plt.clf()
        for kind, data in result.items():
            for cust_grp, item_lvl_cd, date, pred, sale in zip(
                    data['cust_grp_cd'], data[item_lvl_col], data['start_week_day'], data['pred'], data['sales']
            ):

                filtered = sales[(sales['cust_grp_cd'] == cust_grp) & (sales[item_lvl_col] == item_lvl_cd)]
                filtered = filtered[['start_week_day', 'sales']]
                filtered = self.fill_missing_date(df=filtered, col='start_week_day')

                fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
                # fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                ax1.plot(filtered['start_week_day'], filtered['sales'],
                         linewidth=1.3, color='grey', label='history')

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
                    plt.title(f'Date: {date_str} / Cust Group: {cust_grp} / Brand: {self.item_code_map[item_lvl_cd]}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        self.root_path, self.data_vrsn_cd, 'plot', self.data_vrsn_cd + '_' + self.division + '_' +
                        kind + '_' + str(cust_grp) + '_' + str(item_lvl_cd) + '.png'))

                    plt.close(fig)

    @staticmethod
    def connect_dot(hist: pd.DataFrame, date, value) -> pd.DataFrame:
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
