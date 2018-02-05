import datetime
import dateutil.parser
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json

def to_date(datestring):
    date = dateutil.parser.parse(datestring)
    return date

ticks = ['BTC','ETH','LTC','EOS','ADA']
coinmarketcap_hist = {}
date_list = []
start_date = 20130428
end_date = datetime.datetime.now()
iterations = str((end_date-to_date(str(start_date)))//7)
iterations = re.split(' ',iterations)
iterations = iterations[0]
end_date = str(end_date).replace('-','').replace(':','')
end_date = re.split(' ',end_date)
end_date = int(end_date[0])

for i in range(int(iterations)+1):
    date = to_date(str(start_date)) + datetime.timedelta(7*i)
    date = str(date).replace('-','')
    date = re.split(' ',date)
    date = date[0]
    date_list.append(date)

url_a = 'https://coinmarketcap.com/historical/'
for date in date_list:
    url = url_a + date
    response=requests.get(url)
    page=response.text
    soup=BeautifulSoup(page,"lxml")
    tables=soup.find_all("table")
    rows=[row for row in tables[0].find_all('tr')]

    df = pd.read_html(tables[0].prettify())[0]
    df = df[['Symbol','Market Cap']]
    df = df.dropna() # filter out question marks; change question marks to None
    date_data = {}
    rows = len(df)
    for row in range(rows):
            if df['Symbol'][row] in ticks:
                symbol = df['Symbol'][row]
                mkt_cap = str(df['Market Cap'][row]).replace('$','').replace(',','')
                date_data[symbol] = mkt_cap
    mktcap=soup.find_all("span", class_="market-cap")
    mktcap = re.split(' ',str(mktcap).replace('$','').replace(',',''))[5]
    date_data['Crypto Market Cap'] = mktcap

    coinmarketcap_hist[date] = date_data

with open('data/coinmarketcap_hist2.json', 'w') as fp:
    json.dump(coinmarketcap_hist, fp)
print('data/coinmarketcap_hist1.json has been updated!')
