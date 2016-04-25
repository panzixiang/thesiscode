from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

def main():
    test_label = [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + \
               [6]*100 + [7]*100 + [8]*100 

    filename = sys.argv[1]
    

    with open(filename, 'rU') as f:           
        pred = [rec for rec in csv.reader(f, delimiter=',')]
    pred = sum(pred,[])
    pred = [int(x) for x in pred] 
    print zero_one_loss(pred, test_label)   
    cm = confusion_matrix(test_label, pred, labels =[1, 2, 3, 4, 5, 6, 7, 8])
    np.set_printoptions(precision=2)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plot_confusion_matrix(cm)
    plt.show()
    #plt.imasve('smv_confusion', cm, cmap=plt.cm.viridis)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.gist_heat):
    hfont = {'fontname':'Helvetica'}
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    plt.title(title, fontsize=20, **hfont)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.xticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math','physics', 'quant', 'stat'], rotation=45, fontsize=16, **hfont)
    plt.yticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math','physics', 'quant', 'stat'], fontsize=16, **hfont)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16, **hfont)
    plt.xlabel('Predicted label', fontsize=16, **hfont)

if __name__ == "__main__":
    main()     