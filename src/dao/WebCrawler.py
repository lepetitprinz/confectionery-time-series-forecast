import os
import re
import time
import pandas as pd
from selenium import webdriver


class WebCrawler(object):
    css_selector_select_location = "div.page-column-1 > div.content-module > div.locations-list.content-module > a:nth-child(1)"
    css_selector_get_year = "div.monthly-dropdowns > div:nth-child(2) > div.map-dropdown-toggle > h2"
    css_selector_get_month = "div.monthly-dropdowns > div:nth-child(1) > div.map-dropdown-toggle > h2"
    css_selector_get_weather = "div.icon-container > svg"
    css_selector_move_to_month = "div.subnav-items > a[data-gaid='monthly']"
    css_selector_move_to_next_month = "div.two-column-page-content > div.page-column-1 > div.content-module > " \
                                      "div.more-cta-links > a.cta-link"

    def __init__(self, path: dict, url: str):
        self.path_driver = path['path_driver']
        self.path_save = path['path_save']
        self.url = url
        self.monthly_range = 3
        self.sleep_time = 1
        self.ad_frame = 10
        self.wea_col = ['year', 'month', 'day', 'temp_low', 'temp_high', 'weather']

    def init(self):
        driver = webdriver.Chrome(executable_path=self.path_driver)

        # Move to destination url
        driver.get(url=self.url)
        time.sleep(2)

        return driver

    def crawling(self, driver: webdriver, area):
        # Search the necessary location
        driver = self.search_location(driver=driver, area=area)

        driver = self.select_location(driver=driver)

        driver = self.move_to_month(driver=driver)

        wdf = pd.DataFrame({'year': [], 'month': [], 'day': [], 'weather': [], 'temp_high': [],
                            'temp_low': []})
        # wdf = pd.DataFrame()

        n = 0
        while n <= 3:
            year = []
            month = []
            day = []
            weather_list = []
            temp_high_list = []
            temp_low_list = []

            # Get the year data
            data_year = self.get_data_year(driver=driver)

            # Get the month data
            data_month = self.get_data_month(driver=driver)

            # 해당 월 이미 존재하면 다음 달로 넘기도록
            if data_month in set(wdf.month) and n < self.monthly_range:
                driver.find_element_by_css_selector(css_selector=self.css_selector_move_to_next_month).click()
                continue

            # 지난 날짜 제외 해당 월의 첫 날부터 마지막 날까지 dataset
            month_data = driver.find_elements_by_css_selector("a.monthly-daypanel:not(.is-past)")
            for i in range(len(month_data)):
                data_day = int(month_data[i].find_element_by_css_selector("div.date").text)
                if data_day == 1 and len(month_data) >= 28 and len(day) < 7:  # 해당 월의 시작 일이면 초기화
                    year = []
                    month = []
                    day = []
                    weather_list = []
                    temp_high_list = []
                    temp_low_list = []
                if (i > 0 and len(day) > 0) and data_day < day[-1]:  # 다음 월의 데이터 제외하기 위해
                    break

                # Get the weather data
                data_weather, temp_low, temp_high = self.get_data_weather(month=month_data, iter=i)

                year.append(data_year)
                month.append(data_month)
                day.append(data_day)
                weather_list.append(data_weather)
                temp_high_list.append(temp_high)
                temp_low_list.append(temp_low)

            weather_data = {'year': year, 'month': month, 'day': day, 'weather': weather_list,
                            'temp_high': temp_high_list, 'temp_low': temp_low_list}
            n += 1
            if n <= self.monthly_range:
                try:
                    driver.find_element_by_css_selector(self.css_selector_move_to_next_month).click()
                except ConnectionError:
                    driver.refresh()
                    try:
                        driver.find_element_by_css_selector(self.css_selector_move_to_next_month).click()
                    except ConnectionError:
                        pass

            weather_df = pd.DataFrame(weather_data)
            wdf = pd.concat([wdf, weather_df], ignore_index=True)

        # 날씨 데이터 저장
        self.save_result(data=wdf, area=area)

        # Quit the chrome driver
        driver.quit()

    def get_data_year(self, driver: webdriver) -> int:
        # 해당 월의 년도 추출
        try:
            year = int(''.join(re.findall("\d+", driver.find_elements_by_css_selector(
                css_selector=self.css_selector_get_year)[0].text)))

        except ConnectionError:
            driver.refresh()
            driver.find_element_by_css_selector(self.css_selector_move_to_month).click()
            year = int(''.join(re.findall("\d+", driver.find_elements_by_css_selector(
                css_selector=self.css_selector_get_year)[0].text)))

        return year

    def get_data_month(self, driver: webdriver) -> int:
        data_month = int(''.join(re.findall("\d+", driver.find_elements_by_css_selector(
            self.css_selector_get_month)[0].text)))

        return data_month

    # Search the necessary location
    def search_location(self, driver, area) -> webdriver:
        driver.find_element_by_css_selector("div.searchbar-content > form > input").send_keys(area)
        driver.find_element_by_css_selector("div.searchbar-content > svg.icon-search.search-icon").click()
        time.sleep(self.sleep_time)

        return driver

    # Select the first area to be searched
    def select_location(self, driver) -> webdriver:
        driver.find_element_by_css_selector(css_selector=self.css_selector_select_location).click()
        time.sleep(self.sleep_time)

        if self.check_ad(driver=driver):
            driver.refresh()
            driver.find_element_by_css_selector(css_selector=self.css_selector_select_location).click()

        return driver

    # Select the month data
    def move_to_month(self, driver) -> webdriver:
        driver.find_element_by_css_selector(css_selector=self.css_selector_move_to_month).click()
        time.sleep(self.sleep_time)

        return driver

    def get_data_weather(self, month, iter: int):
        # 날씨 정보
        weather = int(''.join(re.findall("\d+", month[iter].find_element_by_css_selector(
            self.css_selector_get_weather).get_attribute('src'))))
        # Minimum temperature
        temp_low = int(''.join(re.findall("\d+", month[iter].find_element_by_css_selector("div.low").text)))

        # Maximum temperature
        temp_high = int(''.join(re.findall("\d+", month[iter].find_element_by_css_selector("div.high").text)))

        return weather, temp_low, temp_high

    def save_result(self, data: pd.DataFrame, area: str) -> None:
        data.to_csv(os.path.join(self.path_save, f'weather_{area}.csv'), index=False)

    def check_ad(self, driver):
        all_iframe = driver.find_elements_by_tag_name("iframe")
        if len(all_iframe) > self.ad_frame:
            print("Ad Found\n")
            # driver.execute_script("""
            #     var elems = document.getElementsByTagName("iframe");
            #     for(var i = 0, max = elems.length; i < max; i++)
            #          {
            #              elems[i].hidden=true;
            #          }
            #                       """)
            return True
        else:
            print('No Ad found')
            return False