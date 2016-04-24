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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
  doc_set = arxiv_11['astro'] + arxiv_11['cond'] + \
            arxiv_11['cs'] + arxiv_11['hep'] + \
            arxiv_11['math'] + arxiv_11['physics'] + \
            arxiv_11['qbio'] + arxiv_11['qfin'] + \
            arxiv_11['quant'] + arxiv_11['stat'] 
  label_set = [1]*len(arxiv_11['astro']) + [2]*len(arxiv_11['cond']) + \
              [3]*len(arxiv_11['cs']) + [4]*len(arxiv_11['hep']) + \
              [5]*len(arxiv_11['math']) + [6]*len(arxiv_11['physics']) + \
              [7]*len(arxiv_11['qbio']) + [8]*len(arxiv_11['qfin']) + \
              [9]*len(arxiv_11['quant']) + [10]*len(arxiv_11['stat']) 

  # list for tokenized documents in loop
  texts = tokenize(doc_set)

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
  test_set = arxiv_12['astro'][0:99] + arxiv_12['cond'][0:99] + \
            arxiv_12['cs'][0:99] + arxiv_12['hep'][0:99] + \
            arxiv_12['math'][0:99] + arxiv_12['physics'][0:99] + \
            arxiv_12['qbio'][0:99] + arxiv_12['qfin'][0:99] + \
            arxiv_12['quant'][0:99] + arxiv_12['stat'][0:99] 
  print "test_set length : " + str(len(test_set))          
  test_label = [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + \
               [6]*100 + [7]*100 + [8]*100 + [9]*100 + [10]*100  
  print "test_label length : " + str(len(test_label))                
  test_texts = tokenize(test_set)

  # build test features
  testPropArray = np.zeros((1000, num_topics))
  for i in range(len(test_texts)):
      test = test_texts[i]
      testProp = ldamodel[dictionary.doc2bow(test)]
      for pair in testProp:
          topicIdx = pair[0]
          weight = pair[1]
          testPropArray[i, topicIdx] = weight

  # all testing
  X_train, X_test, y_train, y_test = topicPropArray, testPropArray, label_set, test_label

  print "training_array length: " + str(len(topicPropArray))
  print "test_array length: " + str(len(testPropArray))
  print "training_label length: " + str(len(label_set)) 
  print "test_label length: " + str(len(test_label))
  print '--------------------------------'
  '''
  # knn3
  knn3 = KNeighborsClassifier(n_neighbors=3)
  knn3.fit(X_train, y_train)
  predictions = knn3.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('knn3pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('knn3cm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'knn3'
  print zero_one_loss(predictions, y_test)
  print '--------------------------------'

  # knn5
  knn5 = KNeighborsClassifier(n_neighbors=5)
  knn5.fit(X_train, y_train)
  predictions = knn5.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('knn5pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('knn5cm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'knn5'
  print zero_one_loss(predictions, y_test)
  print '--------------------------------'

  # svmlin
  svmlin = svm.SVC(kernel='linear')
  svmlin.fit(X_train, y_train)
  predictions = svmlin.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('svmlinpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('svmlincm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'svmlin'
  print zero_one_loss(predictions, y_test)
  print '--------------------------------'

  # Compute confusion matrix
  #cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  #np.savetxt('cm-lin.csv', cm, delimiter=',')
  '''
  np.set_printoptions(precision=2)
  plt.figure()
  plot_confusion_matrix(cm)
  #plt.show()
  plt.imasve('smv_confusion', cm, cmap=plt.cm.viridis)
  '''

  # gnb
  gnb = GaussianNB()
  gnb.fit(X_train, y_train)
  predictions = gnb.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('gnbpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('gnbcm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'gnb'
  print zero_one_loss(predictions, y_test)
  print '--------------------------------'

  # rf50
  rf50 = RandomForestClassifier(n_estimators=50)
  rf50.fit(X_train, y_train)
  predictions = rf50.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('rf50pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('rf50cm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'rf50'
  print zero_one_loss(predictions, y_test)
  print '--------------------------------'
  '''

  # dtree ada
  ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
    n_estimators=400,
    learning_rate=1,
    algorithm="SAMME",
    random_state=None)
    n_estimators = 400
  ada.fit(X_train, y_train)
  predictions = ada.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
  np.savetxt('adapred.csv', predictions.astype(int), fmt='%i', delimiter=",")
  np.savetxt('adacm.txt', cm.astype(int), fmt='%i', delimiter=",")
  # print predictions
  print 'ada'
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