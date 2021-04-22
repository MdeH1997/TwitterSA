import pandas as pd
from pattern.nl import parse, pprint, tag, sentiment, ngrams
from pattern.vector import Document, NB, BINARY, KNN, chngrams
import re
import xml.etree.ElementTree as et

path = #insert your path
###################################################################
#set up De Smedt & Daelemans
xtree = et.parse(".xml")#insert path to dutch lexicon
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
df_train = pd.read_csv(path+'\\filename_traindata.csv',header=0,sep=',', encoding = 'utf-8')#insert filename training data

###################################################################
#train classifier
def plain(clas):
    for index, row in df_train.iterrows():
        v = Document(row['dutch'],type=int(row['sent_num']), stopwords=True)
        clas.train(v)
    for index, row in df_test.iterrows():
        if (row['lex_senti'] == 1)|(row['lex_senti'] == (-1)):
            w = Document(row['dutch'],type=int(row['man_sent']), stopwords=True)
            clas.train(w)
    return clas

def ngram(ngram, clas):
    for index, row in df_train.iterrows():
        v = Document(ngrams(row['dutch'], n=ngram),type=int(row['sent_num']), stopwords=True)
        clas.train(v)
    for index, row in df_test.iterrows():
        if (row['lex_senti'] == 1)|(row['lex_senti'] == (-1)):
            w = Document(ngrams(row['dutch'], n=ngram),type=int(row['man_sent']), stopwords=True)
            clas.train(w)
    return clas

def chngram(ngram, clas):
    for index, row in df_train.iterrows():
        v = Document(chngrams(row['dutch'].lower(), n=ngram),type=int(row['sent_num']), stopwords=True)
        clas.train(v)
    for index, row in df_test.iterrows():
        if (row['lex_senti'] == 1)|(row['lex_senti'] == (-1)):
            w = Document(chngrams(row['dutch'].lower(), n=ngram),type=int(row['man_sent']), stopwords=True)
            clas.train(w)
    return clas
    
###################################################################
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
            senti.append(float(x))
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
NB = NB()
KNN = KNN()
n = [1,2,3,4,5]
df_test['lex_senti'] = lex_sentiment
threshold = [float(-0.2), float(-0.15), float(-0.1), float(-0.05), float(0), float(0.05), float(0.1),float(0.15), float(0.2), float(0.25), float(0.3)]
NB_plain = plain(NB)
KNN_plain = plain(KNN)

for t in threshold:
    lex_sentiment_r = []
    for index, row in df_test.iterrows():
        if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
            lex_sentiment_r.append(1)
        elif row['lex_senti'] < float(t):
            lex_sentiment_r.append(-1)
        elif row['lex_senti'] == 999:
            x = Document(str(row['dutch']), stopwords=True)
            lex_sentiment_r.append(NB_plain.classify(x))
        else:
            lex_sentiment_r.append(999)
    df_test['lex_rounded'] = lex_sentiment_r
    print('NB plain: '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))

    
    for i in n:
        NB_ngram = ngram(i, NB)
        lex_sentiment_r = []
        for index, row in df_test.iterrows():
            if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
                lex_sentiment_r.append(1)
            elif row['lex_senti'] < float(t):
                lex_sentiment_r.append(-1)
            elif row['lex_senti'] == 999:
                doc = ngrams(str(row['dutch']), n=i)
                x = Document(doc)
                lex_sentiment_r.append(NB_ngram.classify(x))
            else:
                lex_sentiment_r.append(999)
    
        df_test['lex_rounded'] = lex_sentiment_r
        print('NB ngram '+str(i)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))

    for i in n:
        NB_chngram = chngram(i, NB)
        lex_sentiment_r = []
        for index, row in df_test.iterrows():
            if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
                lex_sentiment_r.append(1)
            elif row['lex_senti'] < float(t):
                lex_sentiment_r.append(-1)
            elif row['lex_senti'] == 999:
                doc = chngrams(row['dutch'].lower(), n=i)
                x = Document(doc)
                lex_sentiment_r.append(NB_chngram.classify(x))
            else:
                lex_sentiment_r.append(999)
    
        df_test['lex_rounded'] = lex_sentiment_r
        print('NB chngram '+str(i)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))

    lex_sentiment_r = []
    for index, row in df_test.iterrows():
        if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
            lex_sentiment_r.append(1)
        elif row['lex_senti'] < float(t):
            lex_sentiment_r.append(-1)
        elif row['lex_senti'] == 999:
            x = Document(str(row['dutch']), stopwords=True)
            lex_sentiment_r.append(KNN_plain.classify(x))
        else:
            lex_sentiment_r.append(999)
    df_test['lex_rounded'] = lex_sentiment_r
    print('KNN plain: '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))

    for i in n:
        KNN_ngram = ngram(i, KNN)
        lex_sentiment_r = []
        for index, row in df_test.iterrows():
            if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
                lex_sentiment_r.append(1)
            elif row['lex_senti'] < float(t):
                lex_sentiment_r.append(-1)
            elif row['lex_senti'] == 999:
                doc = ngrams(str(row['dutch']), n=i)
                x = Document(doc)
                lex_sentiment_r.append(KNN_ngram.classify(x))
            else:
                lex_sentiment_r.append(999)
        df_test['lex_rounded'] = lex_sentiment_r
        print('KNN ngram '+str(i)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))

    for i in n:
        KNN_chngram = chngram(i, KNN)
        lex_sentiment_r = []
        for index, row in df_test.iterrows():
            if (row['lex_senti'] >= float(t)) & (row['lex_senti'] < 1):
                lex_sentiment_r.append(1)
            elif row['lex_senti'] < float(t):
                lex_sentiment_r.append(-1)
            elif row['lex_senti'] == 999:
                doc = chngrams(row['dutch'].lower(), n=i)
                x = Document(doc)
                lex_sentiment_r.append(KNN_chngram.classify(x))
            else:
                lex_sentiment_r.append(999)
        df_test['lex_rounded'] = lex_sentiment_r
        print('KNN chngram '+str(i)+': '+str(count_test(df_test)[0])+', '+str(count_test(df_test)[1])+', '+str(count_test(df_test)[2])+', '+str(count_test(df_test)[3]))
