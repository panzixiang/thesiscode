from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import gensim
import pickle
import csv
import sys
import numpy as np
import scipy

def main():
    # load pickle
    arxiv_11 = pickle.load(open("2011_big_pop.p", "rb"))
    arxiv_12 = pickle.load(open("2012_big_pop.p", "rb"))
    topiclists = pickle.load(open("minor_subjects.p", "rb"))

    print "loaded pickles"

    # build doc set
    # build doc set
    doc_set = arxiv_11['astro'] + arxiv_11['cond'] + \
            arxiv_11['cs'] + arxiv_11['hep'] + \
            arxiv_11['math'] + arxiv_11['physics'] + \
            arxiv_11['quant'] + arxiv_11['stat'] 
    label_set = [1]*len(arxiv_11['astro']) + [2]*len(arxiv_11['cond']) + \
              [3]*len(arxiv_11['cs']) + [4]*len(arxiv_11['hep']) + \
              [5]*len(arxiv_11['math']) + [6]*len(arxiv_11['physics']) + \
              [7]*len(arxiv_11['quant']) + [8]*len(arxiv_11['stat']) 

    doc_texts = tokenize(doc_set)

    # build indiv training sets
    topic_superset = []
    topic_superset.append(arxiv_11['astro'])
    topic_superset.append(arxiv_11['cond'])
    topic_superset.append(arxiv_11['cs'])
    topic_superset.append(arxiv_11['hep'])
    topic_superset.append(arxiv_11['math'])
    topic_superset.append(arxiv_11['physics'])
    topic_superset.append(arxiv_11['quant'])
    topic_superset.append(arxiv_11['stat'])

    # build individual lda
    lda_superset = []
    num_topics_list = []
    dictionary_set = []

    for topic_set in topic_superset:
        topic_texts = tokenize(topic_set)

        # turn our tokenized documents into a id - term dictionary
        dictionary = corpora.Dictionary(topic_texts)
        dictionary_set.append(dictionary)
            
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in topic_texts]

        # generate LDA model
        num_topics = math.floor(len(topic_set)/100)
        num_topics_list.append(num_topics)
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)
        lda_superset.append(ldamodel)

    print "all LDA built"    

    # build training matrix
    prop_array_superset = []
    for i in range(len(num_topics_list)):
        num_topics = num_topics_list[i]
        topicPropArray = np.zeros((len(doc_texts), num_topics))
        for j in range(len(doc_texts)):
            text = doc_texts[j]
            textProp = lda_superset[i][dictionary_set[i].doc2bow(text)]
            for pair in textProp:
                topicIdx = pair[0]
                weight = pair[1]
                topicPropArray[j, topicIdx] = weight
        prop_array_superset.append(topicPropArray)        

    # concat full feature array
    trainingArray = prop_array_superset[0]
    for i in range(len(prop_array_superset)):
        if i != 0:
            trainingArray = np.concatenate((trainingArray, prop_array_superset[i]), axis = 1)


    print "training matrix built"
    print "------------------"
    print "testing"

    # test on new data
    test_set = arxiv_12['astro'][0:99] + arxiv_12['cond'][0:99] + \
                arxiv_12['cs'][0:99] + arxiv_12['hep'][0:99] + \
                arxiv_12['math'][0:99] + arxiv_12['physics'][0:99] + \
                arxiv_12['quant'][0:99] + arxiv_12['stat'][0:99] 
    test_label = [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + \
                   [6]*100 + [7]*100 + [8]*100 

    test_texts = tokenize(test_set)

    # build indiv test prop array
    test_prop_array_superset = []
    for i in range(len(num_topics_list)):
        num_topics = num_topics_list[i]
        testPropArray = np.zeros((800, num_topics))
        for j in range(len(test_texts)):
            test = test_texts[j]
            testProp = lda_superset[i][dictionary_set[i].doc2bow(test)]
            for pair in testProp:
                topicIdx = pair[0]
                weight = pair[1]
                testPropArray[j, topicIdx] = weight
        test_prop_array_superset.append(testPropArray)   

    # concat full test array 
    testArray = test_prop_array_superset[0] 
    for i in range(len(test_prop_array_superset)):
        if i !=0 :
            testArray = np.concatenate((testArray, test_prop_array_superset[i]), axis = 1)

    cla = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = trainingArray, testArray, label_set, test_label

    print "training_array length: " + str(len(topicPropArray))
    print "test_array length: " + str(len(testPropArray))
    print "training_label length: " + str(len(label_set)) 
    print "test_label length: " + str(len(test_label))
    print '--------------------------------'
    
    # ova
    # gnb
    gnb = GaussianNB()
    sgd = SGDClassifier(loss="hinge", penalty="l2")
    cla = OneVsOneClassifier(gnb)
    cla.fit(X_train, y_train)
    predictions = cla.predict(X_test)
    np.savetxt('ecocovosub_pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print 'ecoc gnb'
    print zero_one_loss(predictions, y_test)
    print '--------------------------------'  

    svmlin = svm.SVC(kernel='linear')
    cla = OneVsOneClassifier(svmlin)
    cla.fit(X_train, y_train)
    predictions = cla.predict(X_test)
    np.savetxt('ecocovosubsvm_pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print 'ecoc svm'
    print zero_one_loss(predictions, y_test)
    print '--------------------------------'


def tokenize(doc_set):
    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # create tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    doc_texts = []    
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
        doc_texts.append(stemmed_tokens)

    return doc_texts    

if __name__ == "__main__":
    main()     