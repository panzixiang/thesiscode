from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn import svm
from sklearn.metrics import zero_one_loss
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
            arxiv_11['qbio'] + arxiv_11['qfin'] + \
            arxiv_11['quant'] + arxiv_11['stat'] 
    label_set = [1]*len(arxiv_11['astro']) + [2]*len(arxiv_11['cond']) + \
              [3]*len(arxiv_11['cs']) + [4]*len(arxiv_11['hep']) + \
              [5]*len(arxiv_11['math']) + [6]*len(arxiv_11['physics']) + \
              [7]*len(arxiv_11['qbio']) + [9]*len(arxiv_11['qfin']) + \
              [10]*len(arxiv_11['quant']) + [2]*len(arxiv_11['stat']) 

    doc_texts = tokenize(doc_set)

    # build indiv training sets
    topic_superset = []
    topic_superset.append(arxiv_11['astro'])
    topic_superset.append(arxiv_11['cond'])
    topic_superset.append(arxiv_11['cs'])
    topic_superset.append(arxiv_11['hep'])
    topic_superset.append(arxiv_11['math'])
    topic_superset.append(arxiv_11['physics'])
    topic_superset.append(arxiv_11['qbio'])
    topic_superset.append(arxiv_11['qfin'])
    topic_superset.append(arxiv_11['quant'])
    topic_superset.append(arxiv_11['stats'])

    # build individual lda
    lda_superset = []
    num_topics_list = []

    for topic_set in topic_superset:
        topic_texts = tokenize(topic_set)

        # turn our tokenized documents into a id - term dictionary
        dictionary = corpora.Dictionary(topic_texts)
            
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in math_texts]

        # generate LDA model
        num_topics = math.floor(len(topic_set))
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
            text = doc_texts[i]
            textProp = lda_superset[i][dictionary.doc2bow(text)]
            for pair in textProp:
                topicIdx = pair[0]
                weight = pair[1]
                topicPropArray[i, topicIdx] = weight
        prop_array_superset.append(topicPropArray)        

    # concat full feature array
    for topicPropArray in prop_array_superset
        trainingArray = np.concatenate((trainingArray, topicPropArray), axis = 1)


    print "training matrix built"
    print "------------------"
    print "testing"

    # test on new data
    test_set = arxiv_12['astro'][0:99] + arxiv_12['cond'][0:99] + \
                arxiv_12['cs'][0:99] + arxiv_12['hep'][0:99] + \
                arxiv_12['math'][0:99] + arxiv_12['physics'][0:9] + \
                arxiv_12['qbio'][0:99] + arxiv_12['qfin'][0:99] + \
                arxiv_12['quant'][0:99] + arxiv_12['stat'][0:99] 
    test_label = [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + \
                   [6]*100 + [7]*100 + [8]*100 + [9]*100 + [10]*100  

    test_texts = tokenize(test_set)

    # build indiv test prop array
    test_prop_array_superset = []
    for i in range(len(num_topics_list)):
        num_topics = num_topics_list[i]
        testPropArray = np.zeros((1000, num_topics))
        for j in range(len(test_texts)):
            test = test_texts[i]
            testProp = lda_superset[i][dictionary.doc2bow(test)]
            for pair in testProp:
                topicIdx = pair[0]
                weight = pair[1]
                testPropArray[i, topicIdx] = weight
        test_prop_array_superset.append(testPropArray)   

    # concat full test array  
    for testPropArray in test_prop_array_superset
        testArray = np.concatenate((testArray, testPropArray), axis = 1)

    cla = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = trainingArray, testArray, label_set, test_label
    cla.fit(X_train, y_train)
    predictions = cla.predict(X_test)
    #print predictions
    print zero_one_loss(predictions, y_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()


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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    plt.imshow(cm, interpolation='nearest', cmap='viridis', vmin=0, vmax=100)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math','physics', 'q-bio', 'q-fin', 'quant', 'stat'], rotation=45)
    plt.yticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math','physics', 'q-bio', 'q-fin', 'quant', 'stat'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

if __name__ == "__main__":
    main()     