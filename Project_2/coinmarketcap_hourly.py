import datetime
import dateutil.parser
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time

coinmarketcap_hourly = {}
url = 'https://coinmarketcap.com/all/views/all/'

i = 0
endr = 0 # TODO periodically save by changing last char of file
while i < 1000:
    response=requests.get(url)
    page=response.text
    soup=BeautifulSoup(page,"lxml")
    tables=soup.find_all("table")
    rows=[row for row in tables[0].find_all('tr')]

    df = pd.read_html(tables[0].prettify())[0]
    df = df[['Symbol','Market Cap']]
    df = df.dropna() # filter out question marks; change question marks to None
    hour_data = {}
    rows = len(df)
    hour = str(datetime.datetime.now()).replace('-','').replace(' ','').split(':')[0] + str(datetime.datetime.now()).split(':')[1]
    for row in range(rows):
        # row = row.replace('?','0')
        # row_adj = int(row.replace('$','').replace(',',''))
        symbol = df['Symbol'][row]
        mkt_cap = df['Market Cap'][row] # .replace('$','').replace(',','')
        # price = df['Price'][row]
        hour_data[symbol] = mkt_cap

    coinmarketcap_hourly[hour] = hour_data
    i += 1
    if i%5 ==0:
        endr += 1
    fil = 'coinmarketcap_hourly{}.json'.format(str(endr))
    with open(fil, 'w+') as fp:
        json.dump(coinmarketcap_hourly, fp)
    print('coinmarketcap_hourly.json has been updated!'.format(hour))
    time.sleep(60*5) # sleep for 60 minutes
