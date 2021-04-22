from pattern.vector import Document, NB
from pattern.nl import ngrams
import pandas as pd
import csv

#import data
path = #insert path
df_train = pd.read_csv(path+'\\filename_traindata.csv',header=0,sep=',', encoding = 'utf-8')#insert filename training data
df_test = pd.read_csv(path+'\\filename_testdata.csv',header=0,sep=',', encoding = 'utf-8')#insert filname test data
df1 = pd.read_csv(path+'\\filename_prepdata1.csv',header=0,sep=';', encoding= 'utf-8')#insert filename preprocessed data 1
df2 = pd.read_csv(path+'\\filename_prepdata2.csv',header=0,sep=';', encoding= 'utf-8')#insert filename preprocessed data 1
df3 = pd.read_csv(path+'\\filename_prepdata3.csv',header=0,sep=';', encoding= 'utf-8')#insert filename preprocessed data 1


df = pd.concat([df1,df2,df3], ignore_index = True)
n = 50000#define chunk size
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]


#train classifier
def classifier(ngram, nb):
    for index, row in df_train.iterrows():
        v = Document(ngrams(row['dutch'], n=ngram),type=(row['sent_num']), stopwords=True)
        nb.train(v)
    for index, row in df_test.iterrows():
        x = Document(ngrams(row['dutch'], n=ngram),type=(row['man_senti']), stopwords=True)
        nb.train(x)
    return nb

NB = NB()#define classifier
NB_t = classifier(1, NB)

count=0
header = ['tweet_id','primary_geo_y','primary_geo_x','sentiment']

#classify tweets
number = 1
for x in list_df:
    tweet_id = []
    primary_geo_y = []
    primary_geo_x = []
    sentiment = []
    number +=1
    for index, row in x.iterrows():
        if count%10000==0:
            print(str(count)+' written')

        sentiment.append(NB_t.classify(ngrams(str(row['dutch']),n=1)))
    
        tweet_id.append(row['tweet_id'])
        primary_geo_y.append(row['primary_geo_y'])
        primary_geo_x.append(row['primary_geo_x'])
    
        rowout = zip(tweet_id, primary_geo_y, primary_geo_x, sentiment)
        with open (path + '\\filename_result'+str(number)+'.csv', 'w', newline = '', encoding='utf-8') as outfile:#insert filename result dataset
            writer = csv.writer(outfile, delimiter = ';', quoting = csv.QUOTE_ALL)
            writer.writerow(header)
            for r in rowout:
                writer.writerow(r)
        count +=1
