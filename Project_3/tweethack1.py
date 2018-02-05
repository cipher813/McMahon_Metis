import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import json
import datetime
import time

# Twitter API Information
consumer_key = 'vZ2rePzW00X6Mr2NVbxcRUSMm'
consumer_secret = 'XkT96BYG7RBJKBdKP2uBl2z4WzPVjTbH155bUldCDT1xVdGiVM'
access_token = '66541059-BLlhwk0IE57qhXQux6XjCKOeVy5MC8NxWR6yaoKQa'
access_secret = 'RWiaKbiPXVZwRslwr9KEWd6ELn4MvEn6eCRSo8npkkZ8t'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

breach_words = ['data leak','security breach','information stolen','password stolen','hacker stole']
ddos_words = ['DDos attack','slow internet','network infiltrated','malicious activity','vulnerability exploit','phishing attack']
hijack_words = ['unauthorized access','stolen identity','hacked account']

fil = 'data/tweethack1.json'
sleep_time = 60*5
i = 0

while i < 100000:
    hack_dict = {}
    breach_list = []
    ddos_list = []
    hijack_list = []
    tstmp = str(datetime.datetime.now()).replace('-','').replace(' ','').split(':')[0] + str(datetime.datetime.now()).split(':')[1]
    try:
        # Twitter sentiment anlysis
        for word in breach_words:
            breach_tweets = api.search(word)
            for tweet in breach_tweets:
                analysis = TextBlob(tweet.text)
                sentiment = analysis.sentiment.polarity
                if sentiment < 0:
                    breach_list.append(str(analysis))

        for word in ddos_words:
            ddos_tweets = api.search(word)
            for tweet in ddos_tweets:
                analysis = TextBlob(tweet.text)
                sentiment = analysis.sentiment.polarity
                if sentiment < 0:
                    ddos_list.append(str(analysis))

        for word in hijack_words:
            hijack_tweets = api.search(word)
            for tweet in hijack_tweets:
                analysis = TextBlob(tweet.text)
                sentiment = analysis.sentiment.polarity
                if sentiment < 0:
                    hijack_list.append(str(analysis))

        hack_dict[tstmp] = breach_list, ddos_list, hijack_list
        # print(hack_dict)
        i += 1

        with open(fil, 'a') as fp:
            json.dump(hack_dict, fp, ensure_ascii=False)

        print('{} has been updated at {}.'.format(fil, tstmp))
    except Exception as e:
        print('{} exception has occurred at {}.'.format(fil, tstmp))
        print(e)
        pass

    time.sleep(sleep_time)
