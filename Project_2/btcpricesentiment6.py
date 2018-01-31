import datetime
import dateutil.parser
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time

import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct

# Twitter API Information
consumer_key = 'vZ2rePzW00X6Mr2NVbxcRUSMm'
consumer_secret = 'XkT96BYG7RBJKBdKP2uBl2z4WzPVjTbH155bUldCDT1xVdGiVM'
access_token = '66541059-BLlhwk0IE57qhXQux6XjCKOeVy5MC8NxWR6yaoKQa'
access_secret = 'RWiaKbiPXVZwRslwr9KEWd6ELn4MvEn6eCRSo8npkkZ8t'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

url = 'https://coinmarketcap.com/all/views/all/'

i = 0
BTCTweets = {}
ticks = ['BTC','ETH','LTC','EOS','ADA']
sleep_time = 60*5 # 5 minutes
fil_mc = 'data/btcpricesentiment_mc9.json'
fil_txt = 'data/btcpricesentiment_txt9.json'

while i < 100000:
    btcpricesentiment_mc = {}
    btcpricesentiment_txt = {}
    txt_btc = []
    txt_eth = []
    txt_blk = []

    agg_btc = []
    agg_eth = []
    agg_blk = []
    tstmp = str(datetime.datetime.now()).replace('-','').replace(' ','').split(':')[0] + str(datetime.datetime.now()).split(':')[1]
    try:
        # Twitter sentiment anlysis
        bitcoin_tweets = api.search('bitcoin')
        ethereum_tweets = api.search('ethereum')
        blockchain_tweets = api.search('blockchain')

        for tweet in bitcoin_tweets:
            analysis = TextBlob(tweet.text)
            sentiment = analysis.sentiment.polarity
            if sentiment != 0:
                agg_btc.append(sentiment)
                txt_btc.append([str(analysis),str(sentiment)])

        for tweet in ethereum_tweets:
            analysis = TextBlob(tweet.text)
            sentiment = analysis.sentiment.polarity
            if sentiment != 0:
                agg_eth.append(sentiment)
                txt_eth.append([str(analysis),str(sentiment)])

        for tweet in blockchain_tweets:
            analysis = TextBlob(tweet.text)
            sentiment = analysis.sentiment.polarity
            if sentiment != 0:
                agg_blk.append(sentiment)
                txt_blk.append([str(analysis),str(sentiment)])

        # Coinmarketcap marketcap snapshot
        response=requests.get(url)
        page=response.text
        soup=BeautifulSoup(page,"lxml")
        tables=soup.find_all("table")
        rows=[row for row in tables[0].find_all('tr')]

        df = pd.read_html(tables[0].prettify())[0]
        df = df[['Symbol','Market Cap']]
        df = df.dropna() # filter out question marks; change question marks to None
        hour_data = {}
        tweet_data = {}
        rows = len(df)
        for row in range(rows):
            if df['Symbol'][row] in ticks:
                symbol = df['Symbol'][row]
                mkt_cap = df['Market Cap'][row] # .replace('$','').replace(',','')
                hour_data[symbol] = mkt_cap
        hour_data['bitcoin_S'] = [sum(agg_btc),len(agg_btc)]
        hour_data['ethereum_S'] = [sum(agg_eth),len(agg_eth)]
        hour_data['blockchain_S'] = [sum(agg_blk),len(agg_blk)]
        mktcap=soup.find_all("span", class_="market-cap")
        mktcap = re.split(' ',str(mktcap))[5]
        hour_data['Crypto Market Cap'] = mktcap

        btcpricesentiment_mc[tstmp] = hour_data

        tweet_data['BTC'] = txt_btc
        tweet_data['ETH'] = txt_eth
        tweet_data['Blk'] = txt_blk
        btcpricesentiment_txt[tstmp] = tweet_data
        i += 1

        with open(fil_txt, 'a') as fp:
            json.dump(btcpricesentiment_txt, fp)
        with open(fil_mc, 'a') as fp:
            json.dump(btcpricesentiment_mc, fp)

        print('{} and {} have been updated at {}.'.format(fil_mc,fil_txt, tstmp))
    except Exception as e:
        print('{} or {} exception has occurred at {}.'.format(fil_mc,fil_txt, tstmp))
        print(e)
        pass

    time.sleep(sleep_time)
