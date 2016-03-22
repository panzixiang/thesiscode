import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from random import shuffle

# confusion matrix plotting code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def main():
    
    filenameLB = 'mfcc_lb.csv'
    allsongcat = pickle.load(open('allsongcat.p', 'rb'))
    #hcdf = pickle.load(open('hcdf_fv.p', 'rb'))
    
    with open('mfcc_lb.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = row
     
    # select training and test sets
    '''
    TEidx = np.array(random.sample(range(0,1000), 100))
    
    training = []
    test = []
    
    trainingLB = []
    testLB = []

    # make numpy arrays
    for i in range(1000):
        if i in TEidx:
            test.append(featureDict[i])
            testLB.append(int(labels[i]))
        else:
            training.append(featureDict[i])
            trainingLB.append(int(labels[i]))
        
    # fit with classifier and predict
    X = np.array(training)
    Y = np.array(trainingLB)

    '''
    l=[allsongcat]
    all_feats = combineFeatures(l)
    feats_shuf = []
    labels_shuf = []
    index_shuf = range(len(labels))
    shuffle(index_shuf)
    for i in index_shuf:
        feats_shuf.append(all_feats[i])
        labels_shuf.append(int(labels[i]))


    X = np.array(feats_shuf)
    Y = np.array(labels_shuf)

    kf = KFold(1000, n_folds=3)
    cla = SVR(kernel="linear")
    selector = RFECV(cla, step=1, cv=3)
    selector = selector.fit(X,Y)

    scores = 0.0
    cm_all = np.zeros((10,10), dtype=np.int)
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        #cla.fit(X_train, y_train)
        predictions = selector.predict(X_test)
        scores += zero_one_loss(predictions, y_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, predictions, labels =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.set_printoptions(precision=2)
        #print(cm_all)
        cm_all = np.add(cm_all, cm)
    
    print scores/3
    plt.figure()
    plot_confusion_matrix(cm_all)

    plt.show()
    
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


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop','Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'], rotation=45)
    plt.yticks(tick_marks, ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop','Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    main()
