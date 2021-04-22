from collections import Counter
import dateutil
import pandas as pd
import numpy as np
import csv

path = #insert your path
f = path+'filename.csv'#insert filename geofiltered data
df = pd.read_csv(f, header=0, sep=',', encoding='utf-8')
print(len(df))
agg = df.groupby(df['tweet_date'])
bots = []
count = []
for group, value in agg:
    c = Counter(value['user'])
    #print(c.items())
    for user in c.items():
        if user[1] > 14:
            bots.append(user[0])
            count.append(user[1])

def ifbot(user):
    if user in bots:
        return True
    else:
        return False
botbool = []
for index, value in df.iterrows():
    botbool.append(ifbot(value['user']))

df['bot'] = botbool
df.drop(df[df['bot'] == True].index, inplace = True)
print(len(df))
df.to_csv(path_or_buf = path+'filename.csv',sep = ',',index=False, mode = 'w', encoding = 'utf-8')#insert filename botfiltered data

df.close()
