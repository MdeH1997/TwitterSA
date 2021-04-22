import pandas as pd
from pattern.nl import parse, pprint, tag, sentiment
import re
from collections import Counter
import xml.etree.ElementTree as et

path = #your path
###################################################################
#set up De Smedt & Daelemans
xtree = et.parse("path to the  lexicon.xml")#insert path to lexicon
xroot = xtree.getroot()

lex_cols = ["lex_id","lex_word","lex_senti"]
rows = []

for node in xroot:
    xml_id = node.attrib.get("cornetto_id")
    xml_word = node.attrib.get("form")
    xml_senti = float(node.attrib.get("polarity"))
    rows.append({"lex_id": xml_id, "lex_word": xml_word, "lex_senti": xml_senti})

lexicon = pd.DataFrame(rows, columns = lex_cols)

###################################################################
#import data
df_test = pd.read_csv(path+'\\filename_testdata.csv',header=0,sep=',', encoding = 'utf-8')#insert filename test data
c = Counter(df_test['man_senti'])
n=[1]
for i in n:
    print(c.items())
    break
###################################################################
#dutch: analyse adjectives
#english: analyse noun(n), verb(v), adverb(r) and adjective(a)
#PoS from pattern:
    #noun = NN/NNS
    #adjective = JJ
    #verb = VB
    #adverb = RB

#Make column with lemmata for tweet
def get_lemma_JJ(text):
    parsing = (parse(text, tags = True, lemmata=True, chunks=False, relations=False).split())
    Lemma_list = []
    for x in parsing:
        for word in x:
            if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
                Lemma_list.append(word[2])
    return Lemma_list
    
#Make columns with score per lemma
def get_scores(df_lemma):
    score_list = []
    senti = []
    for index, row in df_lemma.iterrows():
        score = lexicon['lex_senti'].loc[lexicon['lex_word'] == row['lemmata']].tolist()
        try:
            if score[0] > 0.0:
                score_list.append(max(score))
            else:
                score_list.append(min(score))
        except IndexError as e:
            pass
    if not score_list:
        senti.append(999)
    else:
        for x in score_list:
            senti.append(float(x))#[item for sublist in score_list for item in sublist]
    return senti

count = 0

#loop per tweet
lex_sentiment = []
for index, row in df_test.iterrows():
    df_tweet = pd.DataFrame(columns = ['lemmata'])
    try:
        df_tweet['lemmata'] = get_lemma_JJ(row['dutch'])

        if len(df_tweet) != 0:
            senti_sum = sum(get_scores(df_tweet))
            lex_sentiment.append((senti_sum)/float(len(df_tweet)))
        else:
            lex_sentiment.append(999)

    except TypeError as e:
        lex_sentiment.append(999)
        print(str(count) + " " + str(e))
        pass
    count+=1

def count_test(df):
    true_pos = sum(1 for index, row in df.iterrows() if (row['lex_rounded'] == 1)&(row['man_senti']==1))
    true_neg = sum(1 for index, row in df.iterrows() if (row['lex_rounded'] == (-1))&(row['man_senti']==(-1)))
    false_pos = sum(1 for index, row in df.iterrows() if (row['lex_rounded'] == 1)&(row['man_senti']==(-1)))
    false_neg = sum(1 for index, row in  df.iterrows() if (row['lex_rounded'] == (-1))&(row['man_senti']==1))
    return true_pos, false_pos, false_neg, true_neg

df_test['lex_senti'] = lex_sentiment
threshold = [float(-0.2), float(-0.15), float(-0.1), float(-0.05), float(0), float(0.05), float(0.1),float(0.15), float(0.2), float(0.25), float(0.3)]

for t in threshold:
    lex_sentiment_r = []
    for i in lex_sentiment:
        if (i >= float(t)) & (i < 1):
            lex_sentiment_r.append(1)
        elif i < float(t):
            lex_sentiment_r.append(-1)
        else:
            lex_sentiment_r.append(999)

#print(df_test[['dutch','lex_senti']])
    df_test['lex_rounded'] = lex_sentiment_r
    accuracy = sum([count_test(df_test)[0],count_test(df_test)[3]])/sum(count_test(df_test))
    precision = (count_test(df_test)[0])/sum([count_test(df_test)[0], count_test(df_test)[1]])
    recall = (count_test(df_test)[0])/sum([count_test(df_test)[0], count_test(df_test)[2]])
    f1 = (precision*recall)/(precision+recall)*2
    print(str(t)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))
    
    print(str(t)+': '+str(accuracy)+', '+str(precision)+', '+str(recall)+', '+str(f1)+', '+str(sum(count_test(df_test))/235))#print statistics
