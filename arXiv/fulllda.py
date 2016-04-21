from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn import svm
from sklearn.metrics import zero_one_loss
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

    print "loaded pickles"

    # build doc set
    doc_set = arxiv_11['math'] + arxiv_11['astro']
    label_set = [1]*len(arxiv_11['math']) + [2]*len(arxiv_11['astro'])
    doc_texts = tokenize(doc_set)

    # build indiv training sets
    math_set = arxiv_11['math']

    # list for tokenized documents in loop
    math_texts = tokenize(math_set)

    # turn our tokenized documents into a id - term dictionary
    dictionary = corpora.Dictionary(math_texts)
        
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in math_texts]

    # generate LDA model
    num_topics = 120
    mathldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)

    print "LDA math built"

    # now for astro
    astro_set = arxiv_11['astro']

    # list for tokenized documents in loop
    astro_texts = tokenize(astro_set)

    # turn our tokenized documents into a id - term dictionary
    dictionary = corpora.Dictionary(astro_texts)
        
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in astro_texts]

    # generate LDA model
    num_topics = 60
    astroldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)

    print "LDA astro built"


    # print(ldamodel.print_topics(num_topics=2, num_words=3))

    # look at topic proportion of one document
    # print ldamodel[dictionary.doc2bow(texts[0])]

    # build math topic proportion matrix
    mathtopicPropArray = np.zeros((len(doc_texts), 120))
    for i in range(len(doc_texts)):
        text = doc_texts[i]
        textProp = mathldamodel[dictionary.doc2bow(text)]
        for pair in textProp:
            topicIdx = pair[0]
            weight = pair[1]
            mathtopicPropArray[i, topicIdx] = weight

    # build astro topic proportion matrix
    astrotopicPropArray = np.zeros((len(doc_texts), 60))
    for i in range(len(doc_texts)):
        text = doc_texts[i]
        textProp = astroldamodel[dictionary.doc2bow(text)]
        for pair in textProp:
            topicIdx = pair[0]
            weight = pair[1]
            astrotopicPropArray[i, topicIdx] = weight

    #concat full feature array
    topicPropArray = np.concatenate((mathtopicPropArray,astrotopicPropArray), axis = 1)

    # print topicPropArray

    print "matrix built"
    print "------------------"
    print "testing"

    # test on new data
    test_set = arxiv_12['math'][0:99] + arxiv_12['astro'][0:99]
    test_label = [1]*100 + [2]*100

    test_texts = tokenize(test_set)

    # build indiv test prop array
    mathtestPropArray = np.zeros((200, 120))
    for i in range(len(test_texts)):
        test = test_texts[i]
        testProp = mathldamodel[dictionary.doc2bow(test)]
        for pair in testProp:
            topicIdx = pair[0]
            weight = pair[1]
            mathtestPropArray[i, topicIdx] = weight

    astrotestPropArray = np.zeros((200, 60))
    for i in range(len(test_texts)):
        test = test_texts[i]
        testProp = astroldamodel[dictionary.doc2bow(test)]
        for pair in testProp:
            topicIdx = pair[0]
            weight = pair[1]
            astrotestPropArray[i, topicIdx] = weight

    # concatenate test feature vectors        
    testPropArray = np.concatenate((mathtestPropArray,astrotestPropArray), axis = 1)

    cla = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = topicPropArray, testPropArray, label_set, test_label
    cla.fit(X_train, y_train)
    predictions = cla.predict(X_test)
    print predictions
    print zero_one_loss(predictions, y_test)


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