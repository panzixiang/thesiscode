import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from random import shuffle

# confusion matrix plotting code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def main():
    
    filenameLB = 'mfcc_lb.csv'
    allsongcat = pickle.load(open('mfcc_fv.p', 'rb'))
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
        labels_shuf.append(labels[i])


    X = np.array(feats_shuf)
    Y = np.array(labels_shuf)

    kf = KFold(1000, n_folds=10)
    #cla = RandomForestClassifier(n_estimators=50, max_features = 'log2')
    sgd = SGDClassifier(loss="hinge", penalty="l2")
    #cla = svm.SVC(kernel='linear')
    cla = OutputCodeClassifier(sgd,code_size=2, random_state=0)

    cm_all = np.zeros((10,10), dtype=np.int)

    cb = np.zeros((10,20))
    losses = []
   
    with open('ECOC_error.csv','w') as f1:
        wrtest = csv.writer(f1, quoting=csv.QUOTE_NONNUMERIC,lineterminator='\n')
        scores = 0.0
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
            cla.fit(X_train, y_train)
            predictions = cla.predict(X_test)
            loss = zero_one_loss(predictions, y_test)
            losses.append(loss)
            scores += loss
            # print y_test
            # print predictions
                        
            cb = cla.code_book_

            np.savetxt('codebook.csv', cb, delimiter=',')

            # Compute confusion matrix
            cm = confusion_matrix(y_test, predictions, labels =['1', '2', '3', '4', '5','6', '7', '8', '9', '10'])
            np.set_printoptions(precision=2)
            #print(cm_all)
            cm_all = np.add(cm_all, cm)
        
        # make ECOC coding matrix 0-1 binary
        cb[cb <= 0] = 0
        wrtest.writerow(losses)    
    print cb

    print scores/10
    #plt.figure()
    #plot_confusion_matrix(cm_all)

    #plt.show()
    
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
    plt.imshow(cm, interpolation='nearest', cmap='hot', vmin=0, vmax=100)
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
