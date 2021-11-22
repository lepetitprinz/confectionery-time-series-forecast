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
        self.exg_list = ['particulate', 'micro_particulate']
        self.item_cd_list = ['PM10', 'PM25']
        self.location_list = ['seoul', 'busan', 'daegu', 'incheon', 'gwangju', 'daejeon', 'ulsan', 'gyeonggi',
                         'gangwon', 'chungbuk', 'chungnam', 'jeonbuk', 'jeonnam', 'gyeongbuk', 'gyeongnam',
                         'jeju', 'sejong']

    def get_api_dataset(self) -> list:
        data_list = []
        for item_cd in self.item_cd_list:
            num_rows = self.count_date_range(start_date=self.start_date, end_date=self.end_date)
            xml_tree = self.open_url(url=self.url, service_key=self.service_key, page=self.page, num_rows=num_rows,
                                     item_cd=item_cd)
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db = self.conv_data_to_db(data=data)
            data_info = {
                # 'idx_dtl_cd': stn_id,
                'api_start_day': self.start_date,
                'api_end_day': self.end_date
            }

            data_list.append((data_db, data_info))

        return data_list

    @staticmethod
    def open_url(url, service_key, page, num_rows, item_cd) -> ET:
        query_params = f'?serviceKey={service_key}&returnType=xml&numOfRows={num_rows}&pageNo={page}&itemCode=' \
                       f'{item_cd}&dataGubun=DAILY&searchCondition=MONTH'
        # query_params = f'?serviceKey={service_key}&returnType=xml&numOfRows={num_rows}&pageNo={page}&inqBginDt=' \
        #                f'{start_date}&inqEndDt={end_date}&msrstnName={stn_id}'
        response = urlopen(url + query_params).read()
        xml_tree = ET.fromstring(response)

        return xml_tree

    @staticmethod
    def map_xml_tree(xml_tree: ET) -> pd.DataFrame:
        rows = []
        for node in xml_tree[1][0]:
            item = node.find("itemCode").text
            date = node.find("dataTime").text
            seoul = node.find("seoul").text
            busan = node.find("busan").text
            daegu = node.find("daegu").text
            incheon = node.find("incheon").text
            gwangju = node.find("gwangju").text
            daejeon = node.find("daejeon").text
            ulsan = node.find("ulsan").text
            gyeonggi = node.find("gyeonggi").text
            gangwon = node.find("gangwon").text
            chungbuk = node.find("chungbuk").text
            chungnam = node.find("chungnam").text
            jeonbuk = node.find("jeonbuk").text
            jeonnam = node.find("jeonnam").text
            gyeongbuk = node.find("gyeongbuk").text
            gyeongnam = node.find("gyeongnam").text
            jeju = node.find("jeju").text
            sejong = node.find("sejong").text

            rows.append({'item': item, 'date': date,
                         'seoul': seoul, 'busan': busan, 'daegu': daegu, 'incheon': incheon, 'gwangju': gwangju,
                         'daejeon': daejeon, 'ulsan': ulsan, 'gyeonggi': gyeonggi, 'gangwon': gangwon,
                         'chungbuk': chungbuk, 'chungnam': chungnam, 'jeonbuk': jeonbuk, 'jeonnam': jeonnam,
                         'gyeongbuk': gyeongbuk, 'gyeongnam': gyeongnam, 'jeju': jeju, 'sejong': sejong
                         })

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
