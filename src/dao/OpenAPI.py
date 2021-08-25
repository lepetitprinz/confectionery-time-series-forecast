
import common.config as config
from SqlConfig import SqlConfig
from SqlSession import SqlSession

import pandas as pd
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET


class OpenAPI(object):
    def __init__(self):
        self.url = ''
        self.service_key = ''
        self.page = ''
        self.start_date = ''
        self.end_date = ''
        self.stn_list = []

    def init(self):
        # DB Session
        sql_conf = SqlConfig()
        sess = SqlSession()
        sess.init()
        info = sess.select(sql=sql_conf.get_comm_master())
        sess.close()
        info_dict = dict(zip(info['OPTION_CD'], info['OPTION_VAL']))

        # Initialization
        self.url = info_dict['WEA_URL']
        self.service_key = info_dict['WEA_SERVICE_KEY']
        self.page = info_dict['WEA_PAGE']
        self.stn_list = config.STN_LIST

    def get_api_dataset(self) -> pd.DataFrame:
        num_rows = self.count_date_range(start_date=self.start_date, end_date=self.end_date)
        xml_tree = self.open_url(url=self.url, service_key=self.service_key, page=self.page, num_rows=num_rows,
                                 start_date=self.start_date, end_date=self.end_date, stn_id=self.stn_id)
        data = self.map_xml_tree(xml_tree=xml_tree)

        return data

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
