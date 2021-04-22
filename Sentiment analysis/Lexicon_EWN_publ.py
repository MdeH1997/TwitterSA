import pandas as pd
from pattern.en import parse, pprint, tag
import pattern.nl
import re
from collections import Counter

path = #insert path to lexicon
###################################################################
#set up EmoWordNet
EmoWordNet = pd.read_csv(path + '\\Lexicons\\EmoWordNet1.0.txt',header=0,sep=';', encoding = 'utf-8')
EmoWordNet['Lemma'] = EmoWordNet['Lemma#PoS'].str.split('#', 1).str[0]
EmoWordNet['PoS'] = EmoWordNet['Lemma#PoS'].str.split('#', 1).str[1]

neg_columns = ['AFRAID','ANGRY','ANNOYED','SAD']
pos_columns = ['AMUSED','HAPPY','INSPIRED']
EmoWordNet['negative'] = EmoWordNet[neg_columns].max(axis = 1, skipna=True)
EmoWordNet['positive'] = EmoWordNet[pos_columns].max(axis = 1, skipna=True)
max_list = []
for index, row in EmoWordNet.iterrows():
    if row['negative'] > row['positive']:
        max_list.append((row['negative']*(-1)))
    elif row['positive'] >= row['negative']:
        max_list.append(row['positive'])
    else:
        max_list.append(float(0))
EmoWordNet['max_sentiment'] = max_list


###################################################################
#import data
df_test = pd.read_csv(path+'\\filename_testdata.csv',header=0,sep=',', encoding = 'utf-8')#insert your testdata file name

###################################################################
#Make columns with PoS-tags and Lemmata for tweet
def EWN_Lemma_PoS(text):
    parsing = (parse(text, tags = True, chunks = False, relations = False, lemmata = True, tagset = 'UNIVERSAL').split())
    PoS_list = []
    Lemma_list = []
    for x in parsing:
        for word in x:
            if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
                PoS_list.append('a')
                Lemma_list.append(word[2])
            elif word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
                PoS_list.append('v')
                Lemma_list.append(word[2])
            elif word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS':
                PoS_list.append('n')
                Lemma_list.append(word[2])
            elif word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'WRB':
                PoS_list.append('r')
                Lemma_list.append(word[2])
    return PoS_list, Lemma_list
    
#Make columns with negative and positive score per lemma+PoS
def EWN_scores(df_lemma_pos):
    pos_list = []
    neg_list = []
    for index, row in df_lemma_pos.iterrows():
        pos_list.append(EmoWordNet['positive'].loc
                         [(EmoWordNet['Lemma'] == row['lemmata']) &
                           (EmoWordNet['PoS'] == row['PoS'])].tolist())
        neg_list.append(EmoWordNet['negative'].loc
                         [(EmoWordNet['Lemma'] == row['lemmata']) &
                           (EmoWordNet['PoS'] == row['PoS'])].tolist())
    pos_score = [item for sublist in pos_list for item in sublist]
    neg_score = [item for sublist in neg_list for item in sublist]
    return neg_score, pos_score

def EWN_scores_maxi(df_lemma_pos):
    maxi_list = []
    for index, row in df_lemma_pos.iterrows():
        maxi_list.append(EmoWordNet['max_sentiment'].loc
                         [(EmoWordNet['Lemma'] == row['lemmata']) &
                           (EmoWordNet['PoS'] == row['PoS'])].tolist())
    maxi_score = [item for sublist in maxi_list for item in sublist]
    return maxi_score

count = 0

#loop per tweet
EWN_sentiment = []
for index, row in df_test.iterrows():
    df_tweet = pd.DataFrame(columns = ['lemmata','PoS'])
    try:
        df_tweet['lemmata'] = EWN_Lemma_PoS(row['english'])[1]
        df_tweet['PoS'] = EWN_Lemma_PoS(row['english'])[0]

        if len(df_tweet) != 0:
            senti_sum = sum((EWN_scores_maxi(df_tweet)))
            EWN_sentiment.append((senti_sum)/float(len(df_tweet)))
        else:
            EWN_sentiment.append(999)
            print(str(count) + ' words not in lexicon')
    except TypeError as exc:
        EWN_sentiment.append(999)
        print(str(count) + ' '+ str(exc))
        pass
    count+=1
def count_test(df):
    true_pos = sum(1 for index, row in df.iterrows() if (row['EWN_rounded'] == 1)&(row['man_senti']==1))
    true_neg = sum(1 for index, row in df.iterrows() if (row['EWN_rounded'] == (-1))&(row['man_senti']==(-1)))
    false_pos = sum(1 for index, row in df.iterrows() if (row['EWN_rounded'] == 1)&(row['man_senti']==(-1)))
    false_neg = sum(1 for index, row in  df.iterrows() if (row['EWN_rounded'] == (-1))&(row['man_senti']==1))
    return true_pos, false_pos, false_neg, true_neg

df_test['EWN_senti'] = EWN_sentiment
threshold = [float(-0.2), float(-0.15), float(-0.1), float(-0.05), float(0), float(0.05), float(0.1), float(0.15), float(0.2),float(0.25), float(0.3)]
for t in threshold:
    EWN_sentiment_r = []
    for i in EWN_sentiment:
        if (i >= float(t)) & (i < 1):
            EWN_sentiment_r.append(1)
        elif i < float(t):
            EWN_sentiment_r.append(-1)
        else:
            EWN_sentiment_r.append(999)


    df_test['EWN_rounded'] = EWN_sentiment_r
    accuracy = sum([count_test(df_test)[0],count_test(df_test)[3]])/sum(count_test(df_test))
    precision = (count_test(df_test)[0])/sum([count_test(df_test)[0], count_test(df_test)[1]])
    recall = (count_test(df_test)[0])/sum([count_test(df_test)[0], count_test(df_test)[2]])
    f1 = (precision*recall)/(precision+recall)*2
    print(str(t)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))
    print(str(t)+': '+str(accuracy)+', '+str(precision)+', '+str(recall)+', '+str(f1)+', '+str(sum(count_test(df_test))/235))
