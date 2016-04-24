from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

def main():
    test_label = [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + \
               [6]*100 + [7]*100 + [8]*100 + [9]*100 + [10]*100 

    filename = sys.argv[1]
    print filename

    with open(filename, 'rU') as f:           
        pred = [rec for rec in csv.reader(f, delimiter=',')]
    pred = sum(pred,[])
    pred = [int(x) for x in pred]    
    cm = confusion_matrix(test_label, pred, labels =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm)
    plt.show()
    #plt.imasve('smv_confusion', cm, cmap=plt.cm.viridis)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.gist_heat):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
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