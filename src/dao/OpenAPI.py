import common.config as config

import pandas as pd
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPI(object):
    def __init__(self, info: dict):
        self.url = info['wea_url']
        self.service_key = info['wea_service_key']
        self.page = info['wea_page']
        self.start_date = info['rst_start_day']
        self.end_date = info['rst_end_day']
        self.stn_list = config.STN_LIST
        self.exg_list = ['temp_min', 'temp_max', 'temp_avg', 'rhm_min', 'rhm_avg', 'gsr_sum']

    def get_api_dataset(self) -> pd.DataFrame:
        data_list = []
        for stn_id in self.stn_list:
            num_rows = self.count_date_range(start_date=self.start_date, end_date=self.end_date)
            xml_tree = self.open_url(url=self.url, service_key=self.service_key, page=self.page, num_rows=num_rows,
                                     start_date=self.start_date, end_date=self.end_date, stn_id=stn_id)
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db = self.conv_data_to_db(data=data)
            data_list.append(data_db)

        return data_list

    @staticmethod
    def open_url(url, service_key, page, num_rows, start_date, end_date, stn_id) -> ET:
        query_params = f'?serviceKey={service_key}&numOfRows={num_rows}&pageNo={page}&dataCd=ASOS&dateCd=DAY&startDt=' \
                       f'{start_date}&endDt={end_date}&stnIds={stn_id}'
        response = urlopen(url + query_params).read()
        xml_tree = ET.fromstring(response)

        return xml_tree

    @staticmethod
    def map_xml_tree(xml_tree: ET) -> pd.DataFrame:
        rows = []
        for node in xml_tree[1][1]:
            location = node.find("stnId").text
            date = node.find("tm").text
            temp_min = node.find("minTa").text
            temp_max = node.find("maxTa").text
            temp_avg = node.find("avgTa").text
            rhm_min = node.find("minRhm").text
            rhm_avg = node.find("avgRhm").text
            gsr_sum = node.find("sumGsr").text

            rows.append({"location": location, "date": date, "temp_min": temp_min, "temp_max": temp_max,
                         "temp_avg": temp_avg, "rhm_min": rhm_min, "rhm_avg": rhm_avg, "gsr_sum": gsr_sum})

        return pd.DataFrame(rows)

    @staticmethod
    def count_date_range(start_date, end_date) -> int:
        start_date = datetime.strptime(str(start_date), '%Y%m%d')
        end_date = datetime.strptime(str(end_date), '%Y%m%d')
        dates = pd.date_range(start_date, end_date)

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
            data_exg = data_exg.rename(columns={'date': 'yymm',
                                       exg: 'ref_val',
                                       'location': 'idx_dtl_cd'})
            converted_list.append(data_exg)

        return converted_list
