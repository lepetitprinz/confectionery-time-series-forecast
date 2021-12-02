from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import pandas as pd
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPIWeather(object):
    def __init__(self):
        # Class configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.table_nm = 'M4S_O110710'

        # API configuration
        self.info = None
        self.stn_list = [98, 99, 101, 105, 108, 112, 114, 119, 127, 131, 133, 136, 138, 140, 143, 146,
                         152, 155, 156, 159, 162, 165, 174, 177, 184, 192, 232, 236, 253, 257, 279]
        self.stn_avg_list = [108, 112, 133, 143, 152, 156, 159]    # 서울 / 인천 / 대전 / 대구 / 울산 / 광주 / 부산
        self.exg_list = ['temp_min', 'temp_max', 'temp_avg', 'rhm_min', 'rhm_avg', 'gsr_sum', 'rain_sum']

    def get_api_info(self) -> None:
        self.info = self.io.get_dict_from_db(
            sql=self.sql_conf.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

    def get_api_dataset(self) -> list:
        data_list = []
        for stn_id in self.stn_list:
            num_rows = self.count_date_range()
            xml_tree = self.open_url(num_rows=num_rows, stn_id=stn_id)
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db = self.conv_data_to_db(data=data)
            data_info = {
                'idx_dtl_cd': stn_id,
                'api_start_day': self.info['api_start_day'],
                'api_end_day': self.info['api_end_day']
            }

            data_list.append((data_db, data_info))

        return data_list

    def save_avg_weather(self) -> None:
        avg_info = {'api_start_day': self.info['api_start_day'], 'api_end_day': self.info['api_end_day']}
        weather_avg = self.io.get_df_from_db(sql=self.sql_conf.sql_weather_avg(**avg_info))
        stn_info = {
            'idx_dtl_cd': 999,
            'api_start_day': self.info['api_start_day'],
            'api_end_day': self.info['api_end_day']
        }
        for exg in self.exg_list:
            stn_info['idx_cd'] = exg
            temp = weather_avg[weather_avg['idx_cd'] == exg]
            self.io.delete_from_db(self.sql_conf.del_openapi(**stn_info))
            self.io.insert_to_db(df=temp, tb_name=self.table_nm)

    def save_result_on_db(self, data) -> None:
        for stn_data, stn_info in data:
            for exg_data, exg_id in stn_data:
                stn_info['idx_cd'] = exg_id
                self.io.delete_from_db(self.sql_conf.del_openapi(**stn_info))
                self.io.insert_to_db(df=exg_data, tb_name=self.table_nm)

    def open_url(self, num_rows, stn_id) -> ET:
        query_params = f'?serviceKey={self.info["wea_service_key"]}&numOfRows={num_rows}&pageNo=' \
                       f'{self.info["wea_page"]}&dataCd=ASOS&dateCd=DAY&startDt=' \
                       f'{self.info["api_start_day"]}&endDt={self.info["api_end_day"]}&stnIds={stn_id}'
        response = urlopen(self.info['wea_url'] + query_params).read()
        xml_tree = ET.fromstring(response)

        return xml_tree

    @staticmethod
    def map_xml_tree(xml_tree: ET) -> pd.DataFrame:
        rows = []
        for node in xml_tree[1][1]:
            date = node.find("tm").text           # Date
            location = node.find("stnId").text    # Location
            temp_min = node.find("minTa").text    # Minimum Temperature
            temp_max = node.find("maxTa").text    # Maximum Temperature
            temp_avg = node.find("avgTa").text    # Average Temperature
            rhm_min = node.find("minRhm").text    # Minimum Humidity
            rhm_avg = node.find("avgRhm").text    # Average Humidity
            gsr_sum = node.find("sumGsr").text    # Total Insolation
            rain_sum = node.find("sumRn").text    # Total Rain

            rows.append({
                "location": location, "date": date, "temp_min": temp_min, "temp_max": temp_max, "temp_avg": temp_avg,
                "rhm_min": rhm_min, "rhm_avg": rhm_avg, "gsr_sum": gsr_sum, "rain_sum": rain_sum
            })

        return pd.DataFrame(rows)

    def count_date_range(self) -> int:
        start_date = datetime.strptime(str(self.info['api_start_day']), '%Y%m%d')
        end_date = datetime.strptime(str(self.info['api_end_day']), '%Y%m%d')
        dates = pd.date_range(self.info['api_start_day'], self.info['api_end_day'])

        return len(dates)

    def conv_data_to_db(self, data: pd.DataFrame) -> list:
        converted_list = []
        for exg in self.exg_list:
            data_exg = data[['date', 'location', exg]]
            data_exg['date'] = pd.to_datetime(data_exg['date']).dt.strftime('%Y%m%d')
            data_exg['project_cd'] = 'ENT001'
            data_exg['idx_cd'] = exg.upper()
            data_exg['create_user_cd'] = 'SYSTEM'
            data_exg['create_date'] = datetime.now()
            data_exg = data_exg.rename(columns={'date': 'yymm', exg: 'ref_val', 'location': 'idx_dtl_cd'})
            converted_list.append((data_exg, exg.upper()))

        return converted_list
