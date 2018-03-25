from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import urllib.request
import time

base_url = 'http://www.tadpoletunes.com/tunes/'
end_url = '.htm'
abs_filepath = '/home/ubuntu/Music/Tadpole/Celtic/'

slug_list = ['celtic1/celtic','celtic2/celtic2','celtic3/celtic3']

def download_midis(slug_list,base_url, end_url, abs_filepath):
    midi_dict = {}
    for slug in slug_list:
        url = base_url + slug + end_url
        print(url)
        try:
            response = requests.get(url)
            page = response.text
            soup = BeautifulSoup(page,'lxml')
            name_list = []
            midi_list = []

            for e in soup.find_all('b'):
                print(e)
                if '.mid' in str(e):
                    soup.e.prettify()
                    print(e)
                    e = str(e)
                    midi = e.split('"')[1]
                    loc1 = e.find('b>')
                    loc2 = e.find('</')
                    name = e[loc1+2:loc2]
                    print("Midi: {} Name: {}".format(midi,name))
                    name_list.append(name)
                    midi_list.append(midi)

            slug = slug.split('/')[0]
            for i in range(len(midi_list)):

                name = name_list[i]
                midi = midi_list[i]
                song = re.sub(r"[\d )(-/';:]",'',song)
                name = re.sub(r"[ )(-/';:]",'',name)
                song = midi.replace('.mid','')
                url_path = url + slug + midi
                print("Name: {} Song: {} URL: {}".format(name,song,url_path))

                filepath = abs_filepath + "{}-{}.mid".format(name,song)
                urllib.request.urlretrieve(url, filepath)
                r = requests.get(url)

                midi_dict[song] = {'Name':name, 'URL': url}

                print("{}-{}-{}".format(name,song,url)) # song, artist, url
        except Exception as e:
            print(e)
            pass
        sleep_time = 60*5
        print("Sleeping {} seconds".format(sleep_time))
        time.sleep(sleep_time)
    return midi_dict

midi_dict = download_midis(slug_list, base_url, end_url, abs_filepath)
print("Scraping Complete.")
print(midi_dict)
