from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import urllib.request
import time

base_url = 'http://www.midiworld.com/search/'
end_url = '/?q=pop'
abs_filepath = '/media/cipher000/DATA/Music/MidiWorld/'

start_page = 0
end_page = 100

def download_midis(start_page, end_page, base_url, end_url, abs_filepath):
    midi_dict = {}
    for num in range(start_page,end_page):
        url = base_url + str(num) + end_url
        try:
            response = requests.get(url)
        except Exception as e:
            print(e)
            pass
        page = response.text
        soup = BeautifulSoup(page,'lxml')

        name_list = []

        for e in soup.find_all('li'):
            if 'download' in str(e):
        #         url = soup.find('a',target='_blank')['href']
                e = str(e)
                beg_loc = e.find(' ')
                end_loc = e.find(' - ')
                name = e[beg_loc:end_loc]
                name_list.append(name)

        url_list = []

        midi = soup.find_all('a',target='_blank')

        for link in midi:
            url = link.get('href')
            url_list.append(url)

        # print("Name List Length: {} URL List Length: {}".format(len(name_list),len(url_list)))
        i = 0
        for i in range(len(name_list)):
            pop_name = name_list[i]
            pop_url = url_list[i]
        #     print(pop_name)
        #     print(pop_url)
            midi_dict[pop_name] = pop_url
        #     print(i)
            i+=1

        for k,v in midi_dict.items():
            k = re.sub(r"[\d )(-/']",'',k)
            filepath = abs_filepath + k + ".mid"
            urllib.request.urlretrieve (v, filepath)
            r = requests.get(v)
            print("{}: {}".format(k,v))
        time.sleep(60)
    return midi_dict

midi_dict = download_midis(start_page, end_page, base_url, end_url, abs_filepath)
