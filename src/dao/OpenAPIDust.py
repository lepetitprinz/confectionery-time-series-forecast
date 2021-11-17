import common.config as config

import pandas as pd
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPIDust(object):
    def __init__(self, info: dict):
        self.url = info['dust_url']
        self.service_key = info['dust_service_key']
        self.page = info['dust_page']
        self.start_date = info['api_start_day']
        self.end_date = info['api_end_day']
        self.stn_list = ['%EA%B0%95%EB%82%A8%EA%B5%AC']
        self.exg_list = ['particulate', 'micro_particulate']

    def get_api_dataset(self) -> list:
        data_list = []
        for stn_id in self.stn_list:
            num_rows = self.count_date_range(start_date=self.start_date, end_date=self.end_date)
            xml_tree = self.open_url(url=self.url, service_key=self.service_key, page=self.page, num_rows=num_rows,
                                     start_date=self.start_date, end_date=self.end_date, stn_id=stn_id)
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db = self.conv_data_to_db(data=data)
            data_info = {
                'idx_dtl_cd': stn_id,
                'api_start_day': self.start_date,
                'api_end_day': self.end_date
            }

            data_list.append((data_db, data_info))

        return data_list

    @staticmethod
    def open_url(url, service_key, page, num_rows, start_date, end_date, stn_id) -> ET:
        query_params = f'?serviceKey={service_key}&returnType=xml&numOfRows={num_rows}&pageNo={page}&inqBginDt=' \
                       f'{start_date}&inqEndDt={end_date}&msrstnName={stn_id}'
        response = urlopen(url + query_params).read()
        xml_tree = ET.fromstring(response)

        return xml_tree

    @staticmethod
    def map_xml_tree(xml_tree: ET) -> pd.DataFrame:
        rows = []
        for node in xml_tree[1][1]:
            location = node.find("msrstnName").text
            date = node.find("msurDt").text
            particulate = node.find("pm10Value").text
            micro_particulate = node.find("pm25Value").text

            rows.append({"location": location, "date": date,
                         "particulate": particulate, "micro_particulate": micro_particulate})

        return pd.DataFrame(rows)

    @staticmethod
    def count_date_range(start_date, end_date) -> int:
        start_date = datetime.strptime(str(start_date), '%Y%m%d')
        end_date = datetime.strptime(str(end_date), '%Y%m%d')
        dates = pd.date_range(start_date, end_date)

        return len(dates)

    def conv_data_to_db(self, data: pd.DataFrame):
        converted_list = []
        for exg in self.exg_list:
            data_exg = data[['date', 'location', exg]]
            data_exg['date'] = pd.to_datetime(data_exg['date']).dt.strftime('%Y%m%d')
            data_exg['project_cd'] = 'ENT001'
            data_exg['idx_cd'] = exg.upper()
            data_exg['create_user_cd'] = 'SYSTEM'
            data_exg['create_date'] = datetime.now()
            data_exg = data_exg.rename(columns={
                                            'date': 'yymm',
                                            exg: 'ref_val',
                                            'location': 'idx_dtl_cd'})
            converted_list.append((data_exg, exg.upper()))

        return converted_list
