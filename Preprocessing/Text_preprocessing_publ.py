import pandas as pd
import numpy as np
import math
import time
import json
import csv
from datetime import datetime

import emoji
import re

from pygoogletranslation import Translator

#import tweets as dataframe
path = #insert your path

df1 = pd.read_csv(path+'\\file_name1.csv',header=0,sep=',', encoding = 'utf-8').drop(labels = ['tweet_date','user','bot'], axis = 1)#insert own file name
df2 = pd.read_csv(path+'\\file_name2.csv',header=0,sep=',', encoding = 'utf-8').drop(labels = ['tweet_date','user','bot'], axis = 1)#insert own file name
df3 = pd.read_csv(path+'\\file_name3.csv',header=0,sep=',', encoding = 'utf-8').drop(labels = ['tweet_date','user','bot'], axis = 1)#insert own file name

df_conc = pd.concat([df1,df2,df3,df3], ignore_index = True)
header = ['tweet_id','dutch','english','primary_geo_y','primary_geo_x']

###################################################################################
#remove @mentions, #hashtags, emoji's and numbers
def remove_emojis(text):
    return emoji.get_emoji_regexp().sub(u'', text)
def remove_mention(text):
    result = re.sub(r"@\S+", " ", text)
    return result
def remove_hashtag(text):
    result = re.sub(r"#", " ", text)
    return result
def remove_number(text):
    result = ''.join([i for i in text if not i.isdigit()])
    return result
def remove_url(text):
    result = re.sub(r"http\S+", " ", text)
    return result



###################################################################################
#translate lang
def translator(tweets, language, count):
    translator = Translator()
    x = tweets['clean_text']
    time.sleep(0.4)
    try:
        t = (translator.translate(x, dest=language))
        translated = str(t.text)
    except Exception as e:
        translated = '\\empty\\'
        print(str(count) + ' ' + language + ' Exception Error: ' + str(e))
        pass
    except json.decoder.JSONDecodeError as e:
        translated = '\\empty\\'
        print(str(count) + ' ' + language + ' Json Error: ' + str(e) + '. ' + str(x))
        pass
    return translated

###################################################################################
#text en: won't --> will not | can't --> cannot | n't --> not
def en_wont(text):
    result = re.sub(r"won't", "will not", text)
    return result
def en_cant(text):
    result = re.sub(r"can't", "cannot", text)
    return result
def en_nt(text):
    result = re.sub(r"n't", " not", text)
    return result

extended_list_en = []
def extend_english(text1):
    text2 = en_wont(text1)
    text3 = en_cant(text2)
    text4 = en_nt(text3)
    return text4

####################################################################################
#call functions
count = 0    

tweet_id = []
primary_geo_y = []
primary_geo_x = []
translate_list_nl = []
translate_list_nl = []


for index, row in df.iterrows():
    count += 1
    cleantext = []
    text1 = remove_emojis(row['tweet_text'])
    text2 = remove_mention(text1)
    text3 = remove_hashtag(text2)
    text4 = remove_number(text3)
    text5 = remove_url(text4)
    cleantext.append(text5)
    row['clean_text'] = cleantext

    translate_nl = translator(row, 'nl', count)
    translate_list_nl.append(translate_nl)
    row['dutch'] = translate_nl
    
    translate_en = translator(row, 'en', count)
    extended_en = extend_english(translate_en)
    translate_list_en.append(extended_en)
    row['english'] = extended_en

    tweet_id.append(row['tweet_id'])
    primary_geo_y.append(row['primary_geo_y'])
    primary_geo_x.append(row['primary_geo_x'])

    rowout = zip(tweet_id, translate_list_nl,translate_list_en, primary_geo_y, primary_geo_x)
    with open (path + '\\filename.csv', 'w', newline = '', encoding='utf-8') as outfile:#insert own file name
        writer = csv.writer(outfile, delimiter = ';', quoting = csv.QUOTE_ALL)
        writer.writerow(header)
        for r in rowout:
            writer.writerow(r)



