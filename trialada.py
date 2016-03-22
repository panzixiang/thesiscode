import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from random import shuffle
import itertools
import operator
from sklearn.ensemble import RandomForestClassifier



def main(arg):
    global mfcc, chroma, energy, brightness, hcdf, red;
    
    # first arg will be classifier
    if arg[0] == 'knn3':
        cla = KNeighborsClassifier(n_neighbors=3)
    elif arg[0] == 'knn5':
        cla = KNeighborsClassifier(n_neighbors=5)
    elif arg[0] == 'lda':
        cla = LinearDiscriminantAnalysis()
    elif arg[0] == 'qda':
        cla = QuadraticDiscriminantAnalysis()
    elif arg[0] == 'svmLin':
        cla = svm.SVC(kernel='linear')
    elif arg[0] == 'svmRbf':
        cla = svm.SVC(kernel='rbf')
    elif arg[0] == 'svmSig':
        cla = svm.SVC(kernel='sigmoid')
    elif arg[0] == 'gnb': 
        cla = GaussianNB()
    elif arg[0] == 'rforest':
        cla = RandomForestClassifier(n_estimators=50)
    else:
        exit()

    # open all pickles
    mfcc = pickle.load(open('mfcc_fv.p', 'rb'))
    #chroma = pickle.load(open('chroma_fv.p', 'rb'))
    #energy = pickle.load(open('energy_fv.p', 'rb'))
    #brightness = pickle.load(open('brightness_fv.p', 'rb'))
    #hcdf = pickle.load(open('hcdf_fv.p', 'rb'))

    # get labels
    with open('mfcc_lb.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = row

    # build Adaboost
    ada = AdaBoostClassifier(base_estimator=cla, n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=None)        

    # for loop each feature, selecting the two highest
    featureCombos = []
    for i in range(5):
        featureCombos += list(itertools.combinations([mfcc], i+1));
    outcomes = {}

    for l in featureCombos:
        featStr, error = runTrial(ada, arg[0], l, labels)
        outcomes[featStr] = error

    outcomesSorted = sorted(outcomes.items(), key=operator.itemgetter(1))
    
    #print "-------------"
    #print "top five:"
    index = 0
    for o in outcomesSorted:
        if index == 5:
            break
        formatted = "%s" % (o,)
        #print formatted # + ", error=" + str(outcomesSorted(o))
        index += 1


   
def runTrial(cla, claName, featList, labels): 
    feats = combineFeatures(featList)
    feats_shuf = []
    labels_shuf = []
    index_shuf = range(len(labels))
    shuffle(index_shuf)
    for i in index_shuf:
        feats_shuf.append(feats[i])
        labels_shuf.append(labels[i])

    #training, test, trainingLB, testLB = getFeatures(featList, labels)
    X = np.array(feats_shuf)
    Y = np.array(labels_shuf)
    scores = 0.0
    kf = KFold(1000, n_folds=10)
    # for train, test in kf:
    #    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    #    cla.fit(X_train, y_train)
    #    predictions = cla.predict(X_test)
    #    print zero_one_loss(predictions, y_test)
    #    scores += zero_one_loss(predictions, y_test)
    scores = cross_validation.cross_val_score(cla, X, Y, scoring='accuracy', cv=10)

    print claName + "," + printFeatures(featList)
    print 1 - np.mean(scores)
    return printFeatures(featList), 1-np.mean(scores)


def printFeatures(featList):
    featStr = ''
    for f in featList:
        if f is mfcc:
            featStr += 'mfcc'
        elif f is chroma:
            featStr += 'chroma'
        elif f is brightness:
            featStr += 'brightness'
        elif f is energy:
            featStr += 'energy'
        elif f is hcdf:
            featStr += 'hcdf'
        elif f is red:
            featStr += 'red3'
        featStr += ' '

    return featStr[:-1]

def combineFeatures(features):
    l = []
    # make total featuresDict
    featureDict = features[0];
    if len(features) > 1:
        for index in range(1, len(features)):
            for i in range(1000):
                featureDict[i] += features[index][i]

    for i in range(1000):
        l.append(featureDict[i])

    return l

def getFeatures(features, labels):
    # select training and test sets
    TEidx = np.array(random.sample(range(0,1000), 100))
    
    training = []
    test = []
    
    trainingLB = []
    testLB = []

    # make total featuresDict
    featureDict = features[0];
    if len(features) > 1:
        for index in range(1, len(features)):
            for i in range(1000):
                featureDict[i] += features[index][i]

    for i in range(1000):
        if i in TEidx:
            test.append(featureDict[i])
            testLB.append(labels[i])
        else:
            training.append(featureDict[i])
            trainingLB.append(labels[i])

    return training, test, trainingLB, testLB


if __name__ == "__main__":
    main(sys.argv[1:])
