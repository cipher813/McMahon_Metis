import tweepy
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
import datetime
import time

# Twitter API Information
consumer_key = 'vZ2rePzW00X6Mr2NVbxcRUSMm'
consumer_secret = 'XkT96BYG7RBJKBdKP2uBl2z4WzPVjTbH155bUldCDT1xVdGiVM'
access_token = '66541059-BLlhwk0IE57qhXQux6XjCKOeVy5MC8NxWR6yaoKQa'
access_secret = 'RWiaKbiPXVZwRslwr9KEWd6ELn4MvEn6eCRSo8npkkZ8t'

oauth = OAuth(access_token, access_secret, consumer_key, consumer_secret)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

fil = 'data/twitter_trends1.json'
sleep_time = 60*5
i = 0
while i < 100000:
    twitter_trends = {}
    tstmp = str(datetime.datetime.now()).replace('-','').replace(' ','').split(':')[0] + str(datetime.datetime.now()).split(':')[1]

    try:
        trends_list = []
        trends1 = api.trends_place(1)
        data = trends1[0]
        trends = data['trends']
        names = [trend['name'] for trend in trends]
        twitter_trends[tstmp] = names
        i += 1

        with open(fil, 'a') as fp:
            json.dump(twitter_trends, fp, ensure_ascii=False)

        print('{} has been updated at {}.'.format(fil, tstmp))
    except Exception as e:
        print('{} exception has occurred at {}.'.format(fil, tstmp))
        print(e)
        pass

    time.sleep(sleep_time)
