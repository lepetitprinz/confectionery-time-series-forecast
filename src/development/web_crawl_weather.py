import os

from dao.WebCrawler import WebCrawler


path = {
    'path_driver': os.path.join('..', '..', 'crawl', 'chromedriver'),
    'path_save': os.path.join('..', '..', 'crawl'),
}
url = 'https://www.accuweather.com/ko'
area = '서울'

wc = WebCrawler(
    path=path,
    url=url
)

driver = wc.init()
wc.crawling(driver=driver, area=area)
