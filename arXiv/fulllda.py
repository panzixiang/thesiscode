from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
import csv
import sys
import numpy as np
import scipy as sp

tokenizer = RegexpTokenizer(r'\w+')

# load pickle
arxiv_11 = pickle.load(open("2011_big_pop.p", "rb"))
arxiv_12 = pickle.load(open("2011_big_pop.p", "rb"))

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
# doc_a = 'We give necessary and sufficient conditions for the (bounded) law of the iterated logarithm for $U$-statistics in Hilbert spaces. As a tool we also develop moment and tail estimates for canonical Hilbert-space valued$U$-statistics of arbitrary order, which are of independent interest.'
# doc_b = 'Generalization of the Kac integral and Kac method for paths measure based on the Levy distribution has been used to derive fractional diffusion equation. Application to nonlinear fractional Ginzburg-Landau equation is discussed.' 

# build doc set
doc_set = arxiv_11['math']

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
num_topics = 30
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)

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

# test on new data
test_set = arxiv_12['math'][0:9]

# loop through test list
for i in test_set:
    print i
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    test_set.append(stemmed_tokens)

# calculate similarity measure
confidence = []
for test in test_set:
    sim_score = np.zeros(len(texts))
    test_vec = np.zeros(1, num_topics)
    newProp = ldamodel[dictionary.doc2bow(test)]
    for pair in newProp:
        topicIdx = pair[0]
        weight = pair[1]
        test_vec[topicIdx] = weight
    simscore = [scipy.spatial.distance.jaccard(test_vec, row) for row in topicPropArray]
    max_score = np.amax(sim_score)
    print max_score
    confidence.append(max_score)