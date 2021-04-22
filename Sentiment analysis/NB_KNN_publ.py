from pattern.vector import Document, NB, BINARY, KNN, chngrams, SVM, RADIAL, POLYNOMIAL, gridsearch
from pattern.nl import ngrams, parsetree
import pandas as pd

#import data
path = ''#insert your path
df_train = pd.read_csv(path+'\\filename_traindata.csv',header=0,sep=',', encoding = 'utf-8')#insert filename training data
df_test = pd.read_csv(path+'\\filename_testdata.csv',header=0,sep=',', encoding = 'utf-8')#insert filename test data

#train classifier

n = [1,2,3,4,5]
def plain_nl(clas):
    v = [Document(row['dutch'],type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]  
    x = [Document(row['dutch'],type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+' plain dutch: '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))

def ngram_nl(ngram, clas):
    v = [Document(ngrams(row['dutch'], n=ngram),type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]    
    x = [Document(ngrams(row['dutch'], n=ngram),type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+', ngram dutch '+str(ngram)+': '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))

def chngram_nl(ngram, clas):
    v = [Document(chngrams(row['dutch'].lower(), n=ngram),type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]  
    x = [Document(chngrams(row['dutch'].lower(), n=ngram),type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+', chngram dutch '+str(ngram)+': '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))

def plain_en(clas):
    v = [Document(row['english'],type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]
    x = [Document(row['english'],type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+' plain english: '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))

def ngram_en(ngram, clas):
    v = [Document(ngrams(row['english'], n=ngram),type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]
    x = [Document(ngrams(row['english'], n=ngram),type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+', ngram english '+str(ngram)+': '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))

def chngram_en(ngram, clas):
    v = [Document(chngrams(row['english'].lower(), n=ngram),type=int(row['sent_num']), stopwords=True) for index, row in df_train.iterrows()]     
    x = [Document(chngrams(row['english'].lower(), n=ngram),type=int(row['man_senti']), stopwords=True) for index, row in df_test.iterrows()]
    print(str(clas)+', chngram english '+str(ngram)+': '+str(clas.confusion_matrix(x)(True)[0])+', '+str(clas.confusion_matrix(x)(True)[2])+', '+str(clas.confusion_matrix(x)(True)[3])+', '+str(clas.confusion_matrix(x)(True)[1]))


NB = NB()
KNN = KNN(k=15)
n = [1,2,3,4,5]

plain_nl(NB)
for i in n:
    ngram_nl(i,NB)
    chngram_nl(i,NB)
plain_nl(KNN)
for i in n:
    ngram_nl(i,KNN)
    chngram_nl(i,KNN)

plain_en(NB)
for i in n:
    ngram_en(i,NB)
    chngram_en(i,NB)
plain_en(KNN)
for i in n:
    ngram_en(i,KNN)
    chngram_en(i,KNN)
