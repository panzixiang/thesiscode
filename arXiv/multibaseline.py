from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import RandomForestClassifier
import gensim
import pickle
import csv
import sys
import numpy as np
import scipy


tokenizer = RegexpTokenizer(r'\w+')

# load pickle
arxiv_11 = pickle.load(open("2011_big_pop.p", "rb"))
arxiv_12 = pickle.load(open("2012_big_pop.p", "rb"))

print "loaded pickles"

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# build doc set
doc_set = arxiv_11['astro'] + arxiv_11['cond'] + \
          arxiv_11['cs'] + arxiv_11['hep'] + \
          arxiv_11['math'] + arxiv_11['physics'] + \
          arxiv_11['qbio'] + arxiv_11['qfin'] + \
          arxiv_11['quant'] + arxiv_11['stat'] 
label_set = [1]*len(arxiv_11['astro']) + [2]*len(arxiv_11['cond']) + \
            [3]*len(arxiv_11['cs']) + [4]*len(arxiv_11['hep']) + \
            [5]*len(arxiv_11['math']) + [6]*len(arxiv_11['physics']) + \
            [7]*len(arxiv_11['qbio']) + [9]*len(arxiv_11['qfin']) + \
            [10]*len(arxiv_11['quant']) + [2]*len(arxiv_11['stat']) 

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id - term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
num_topics = 460
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)

print "LDA built"

# print(ldamodel.print_topics(num_topics=2, num_words=3))

# look at topic proportion of one document
# print ldamodel[dictionary.doc2bow(texts[0])]

# build topic proportion matrix
topicPropArray = np.zeros((len(texts), num_topics))
for i in range(len(texts)):
    text = texts[i]
    textProp = ldamodel[dictionary.doc2bow(text)]
    for pair in textProp:
        topicIdx = pair[0]
        weight = pair[1]
        topicPropArray[i, topicIdx] = weight

# print topicPropArray

print "matrix built"
print "------------------"
print "testing"

# test on new data
test_set = arxiv_12['astro'][0:9] + arxiv_12['cond'][0:9] + \
          arxiv_12['cs'][0:9] + arxiv_12['hep'][0:9] + \
          arxiv_12['math'][0:9] + arxiv_12['physics'][0:9] + \
          arxiv_12['qbio'][0:9] + arxiv_12['qfin'][0:9] + \
          arxiv_12['quant'][0:9] + arxiv_12['stat'][0:9] 
test_label = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10 + \
             [6]*10 + [7]*10 + [8]*10 + [9]*10 + [10]*10  

test_texts = []

# loop through test list
for i in test_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    test_texts.append(stemmed_tokens)

# calculate similarity measure
confidence = []
testPropArray = np.zeros((20, num_topics))
for i in range(len(test_texts)):
    test = test_texts[i]
    testProp = ldamodel[dictionary.doc2bow(test)]
    for pair in testProp:
        topicIdx = pair[0]
        weight = pair[1]
        testPropArray[i, topicIdx] = weight

# all testing
X_train, X_test, y_train, y_test = topicPropArray, testPropArray, label_set, test_label
# knn3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
predictions = knn3.predict(X_test)
# print predictions
print 'knn3'
print zero_one_loss(predictions, y_test)
print '--------------------------------'

# knn5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
predictions = knn5.predict(X_test)
# print predictions
print 'knn5'
print zero_one_loss(predictions, y_test)
print '--------------------------------'

# svmrbf
svmrbf = svm.SVC(kernel='rbf')
svmrbf.fit(X_train, y_train)
predictions = svmrbf.predict(X_test)
# print predictions
print 'svmrbf'
print zero_one_loss(predictions, y_test)
print '--------------------------------'

# gnb
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
# print predictions
print 'gnb'
print zero_one_loss(predictions, y_test)
print '--------------------------------'

# rf50
rf50 = RandomForestClassifier(n_estimators=50)
rf50.fit(X_train, y_train)
predictions = rf50.predict(X_test)
# print predictions
print 'rf50'
print zero_one_loss(predictions, y_test)
print '--------------------------------'

