import pandas as pd
from collections import Counter
import csv

def get_tweetdata(tweets, count, result):
    header = ['tweet_id','tweet_date','tweet_text','primary_geo_y','primary_geo_x','user']
    tweet_id = []
    tweet_date = []
    tweet_text = []
    coordinates_y = []
    coordinates_x = []
    user = []

    for index, row in tweets.iterrows():
        count += 1
        tweet_id.append(row['id'])
        tweet_date.append(str(row['created_at'])[:10])
        tweet_text.append(row['full_text'])
        user.append(row['user']['name'])
        try:
            if row['coordinates']:
                coordinates_x.append(row['coordinates']['coordinates'][1])
                coordinates_y.append(row['coordinates']['coordinates'][0])
            elif row['place']:
                coordinates_x.append(row['place']['bounding_box']['coordinates'][0][0][1])
                coordinates_y.append(row['place']['bounding_box']['coordinates'][0][0][0])
            else:
                coordinates_x.append('not available')
                coordinates_y.append('not available')
        except TypeError:
            print(str(count)+'error geo')
            pass

    result = zip(tweet_id,tweet_date,tweet_text,coordinates_x,coordinates_y,user)
    return result

path = #insert your path
count = 3000000
rowlength = 0
tweets = pd.read_json(path + "\\filename.csv", orient='records',typ='frame',lines=True, chunksize = 250000)#insert file name hydrated twitter data
header = ['tweet_id','tweet_date','tweet_text','primary_geo_y','primary_geo_x','user']
result = zip()

for chunk in tweets:
    get_tweets = get_tweetdata(chunk, count, result)
    count += 250000
    resfile = path + '\\filename' + str(count) + '.csv'#insert file name geofiltered tweets
    with open(resfile, 'w', newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(header)
        for r in get_tweets:
            writer.writerow(r)
