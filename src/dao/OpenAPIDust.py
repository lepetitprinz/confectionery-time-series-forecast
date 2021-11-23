from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig

import pandas as pd
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPIDust(object):
    def __init__(self):
        # Class configuration
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.table_nm = 'M4S_O110710'

        # API configuration
        self.info = None
        self.item_cd_list = ['PM10', 'PM25']
        self.location_list = ['seoul', 'busan', 'daegu', 'incheon', 'gwangju', 'daejeon', 'ulsan', 'gyeonggi',
                              'gangwon', 'chungbuk', 'chungnam', 'jeonbuk', 'jeonnam', 'gyeongbuk', 'gyeongnam',
                              'jeju', 'sejong']

    def get_api_info(self):
        self.info = self.io.get_dict_from_db(
            sql=self.sql_conf.sql_comm_master(),
            key='OPTION_CD',
            val='OPTION_VAL'
        )

    def get_api_dataset(self) -> list:
        data_list = []
        for item_cd in self.item_cd_list:
            num_rows = self.count_date_range(
                start_date=self.info['api_start_day'],
                end_date=self.info['api_end_day']
            )
            xml_tree = self.open_url(
                url=self.info['dust_url'],
                service_key=self.info['dust_service_key'],
                page=self.info['dust_page'],
                num_rows=num_rows,
                item_cd=item_cd
            )
            data = self.map_xml_tree(xml_tree=xml_tree)
            data_db, from_to_date = self.conv_data_to_db(item=item_cd, data=data)
            data_info = {
                'idx_cd': item_cd,
                'api_start_day': from_to_date['start_day'],
                'api_end_day': from_to_date['end_day']
            }

            data_list.append((data_db, data_info))

        return data_list

    def save_result(self, data):
        for item_data, item_info in data:
            for loc_data, loc_id in item_data:
                loc_data = loc_data.fillna(0.0)
                item_info['idx_dtl_cd'] = loc_id.upper()
                self.io.delete_from_db(self.sql_conf.del_openapi(**item_info))
                self.io.insert_to_db(df=loc_data, tb_name=self.table_nm)

    @staticmethod
    def open_url(url, service_key, page, num_rows, item_cd) -> ET:
        query_params = f'?serviceKey={service_key}&returnType=xml&numOfRows={num_rows}&pageNo={page}&itemCode=' \
                       f'{item_cd}&dataGubun=DAILY&searchCondition=MONTH'
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

    def conv_data_to_db(self, item: str, data: pd.DataFrame):
        converted_list = []
        from_to_date = {}
        for loc in self.location_list:
            # Filtered by location
            filtered = data[['item', 'date', loc]]
            filtered['date'] = pd.to_datetime(filtered['date']).dt.strftime('%Y%m%d')
            from_to_date['start_day'] = filtered['date'].min()
            from_to_date['end_day'] = filtered['date'].max()
            # Add information
            filtered['project_cd'] = 'ENT001'
            filtered['item'] = item.upper()
            filtered['idx_dtl_cd'] = loc.upper()
            filtered['idx_dtl_nm'] = loc.upper()
            filtered['create_user_cd'] = 'SYSTEM'
            filtered['create_date'] = datetime.now()
            filtered = filtered.rename(columns={'date': 'yymm', loc: 'ref_val', 'item': 'idx_cd'})
            converted_list.append((filtered, loc.upper()))

        return converted_list, from_to_date
