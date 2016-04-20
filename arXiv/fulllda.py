from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn import svm
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
doc_set = arxiv_11['math'] + arxiv_11['astro']
label_set = [1]*len(arxiv_11['math']) + [2]*len(arxiv_11['astro'])

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
num_topics = 170
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
test_set = arxiv_12['astro'][0:9] + arxiv_12['math'][0:9]
test_label = [2]*10 + [1]*10 

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


cla = svm.SVC(kernel='linear')
X_train, X_test, y_train, y_test = topicPropArray, testPropArray, label_set, test_label
cla.fit(X_train, y_train)
predictions = cla.predict(X_test)
print zero_one_loss(predictions, y_test)
'''           
    sim_score = [(1-scipy.spatial.distance.cosine(test_vec, row)) for row in topicPropArray]
    max_score = np.amax(sim_score)
    print max_score
    mean_score = np.mean(sim_score)
    print mean_score
    print '\n'
    confidence.append(max_score)
'''