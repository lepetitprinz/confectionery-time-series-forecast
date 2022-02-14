from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import pandas as pd
import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPIWeather(object):
    def __init__(self):
        # Class configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.table_nm = 'M4S_O110710'

        # API configuration
        self.date = {}
        self.info = None
        self.stn_list = [98, 99, 101, 105, 108, 112, 114, 119, 127, 131, 133, 136, 138, 140, 143, 146,
                         152, 155, 156, 159, 162, 165, 174, 177, 184, 192, 232, 236, 253, 257, 279]
        # self.stn_avg_list = [108, 112, 133, 143, 152, 156, 159]    # 서울 / 인천 / 대전 / 대구 / 울산 / 광주 / 부산
        self.exg_list = ['temp_min', 'temp_max', 'temp_avg', 'rhm_min', 'rhm_avg', 'gsr_sum', 'rain_sum']

    # Get API information from DB
    def get_api_info(self) -> None:
        self.info = self.io.get_dict_from_db(
            sql=self.sql_conf.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

    # Set date period information
    def set_date_range(self) -> None:
        today = datetime.date.today()
        prev_monday = today - datetime.timedelta(days=today.weekday() + 8)   # sunday before 2 weeks
        prev_sunday = today - datetime.timedelta(days=today.weekday() + 1)   # Previous sunday

        prev_monday = datetime.date.strftime(prev_monday, '%Y%m%d')
        prev_sunday = datetime.date.strftime(prev_sunday, '%Y%m%d')

        self.date = {'from': prev_monday, 'to': prev_sunday}
        # self.date = {'from': '20190101', 'to': '20191231'}

    # Get the dataset from API call
    def get_api_dataset(self) -> list:
        data_list = []
        for stn_id in self.stn_list:
            num_rows = self.count_date_range()
            xml_tree = self.open_url(num_rows=num_rows, stn_id=stn_id)
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db = self.conv_data_to_db(data=data)
            data_info = {
                'idx_dtl_cd': stn_id,
                'api_start_day': self.date['from'],
                'api_end_day': self.date['to']
            }

            data_list.append((data_db, data_info))

        return data_list

    # Day count
    def count_date_range(self) -> int:
        date_from = self.date['from']
        date_to = self.date['to']
        dates = pd.date_range(date_from, date_to)

        return len(dates)

    # Open API URL
    def open_url(self, num_rows, stn_id) -> ET:
        query_params = f'?serviceKey={self.info["wea_service_key"]}&numOfRows={num_rows}&pageNo=' \
                       f'{self.info["wea_page"]}&dataCd=ASOS&dateCd=DAY&startDt=' \
                       f'{self.date["from"]}&endDt={self.date["to"]}&stnIds={stn_id}'
        response = urlopen(self.info['wea_url'] + query_params).read()
        xml_tree = ET.fromstring(response)

        return xml_tree

    @staticmethod
    def map_xml_tree(xml_tree: ET) -> pd.DataFrame:
        rows = []
        for node in xml_tree[1][1]:
            date = node.find("tm").text           # Date
            loc_cd = node.find("stnId").text      # Location Code
            loc_nm = node.find("stnNm").text      # Location Name
            temp_min = node.find("minTa").text    # Minimum Temperature
            temp_max = node.find("maxTa").text    # Maximum Temperature
            temp_avg = node.find("avgTa").text    # Average Temperature
            rhm_min = node.find("minRhm").text    # Minimum Humidity
            rhm_avg = node.find("avgRhm").text    # Average Humidity
            gsr_sum = node.find("sumGsr").text    # Total Insolation
            rain_sum = node.find("sumRn").text    # Total Rain

            rows.append({
                "loc_cd": loc_cd, "loc_nm": loc_nm, "date": date, "temp_min": temp_min, "temp_max": temp_max,
                "temp_avg": temp_avg, "rhm_min": rhm_min, "rhm_avg": rhm_avg, "gsr_sum": gsr_sum, "rain_sum": rain_sum
            })

        return pd.DataFrame(rows)

    def save_result_on_db(self, data: list) -> None:
        for stn_data, stn_info in data:
            for exg_data, exg_id in stn_data:
                stn_info['idx_cd'] = exg_id
                self.io.delete_from_db(self.sql_conf.del_openapi(**stn_info))
                self.io.insert_to_db(df=exg_data, tb_name=self.table_nm, verbose=False)

    def conv_data_to_db(self, data: pd.DataFrame) -> list:
        # fill na
        data = data.fillna(0)

        converted_list = []
        for exg in self.exg_list:
            data_exg = data[['date', 'loc_cd', 'loc_nm', exg]]
            data_exg['date'] = pd.to_datetime(data_exg['date']).dt.strftime('%Y%m%d')
            data_exg['project_cd'] = 'ENT001'
            data_exg['idx_cd'] = exg.upper()
            data_exg['create_user_cd'] = 'SYSTEM'
            data_exg['create_date'] = datetime.datetime.now()
            data_exg = data_exg.rename(columns={
                'date': 'yymm', exg: 'ref_val', 'loc_cd': 'idx_dtl_cd', 'loc_nm': 'idx_dtl_nm'
            })
            converted_list.append((data_exg, exg.upper()))

        return converted_list

    # def save_avg_weather(self) -> None:
    #     avg_info = {'api_start_day': self.date['from'], 'api_end_day': self.date['to']}
    #     weather_avg = self.io.get_df_from_db(sql=self.sql_conf.sql_weather_avg(**avg_info))
    #     stn_info = {
    #         'idx_dtl_cd': 999,
    #         'api_start_day': self.date['from'],
    #         'api_end_day': self.date['to']
    #     }
    #     for exg in self.exg_list:
    #         stn_info['idx_cd'] = exg
    #         temp = weather_avg[weather_avg['idx_cd'] == exg]
    #         self.io.delete_from_db(self.sql_conf.del_openapi(**stn_info))
    #         self.io.insert_to_db(df=temp, tb_name=self.table_nm)
