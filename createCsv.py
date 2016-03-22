import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import random
import sys

def main(argv):
    fn = 'results' + argv[0] + '.csv'
    with open(fn, 'w') as w:
        writer = csv.writer(w)

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            print line
            split = line.split(',')

            if len(split) < 2:
                break

            new_line = []
            new_line.append(split[0])
            new_line.append(split[1][:-1])
            error = sys.stdin.readline()
            print error
            new_line.append(error[1:-1])
            writer.writerow(new_line)


if __name__ == "__main__":
    main(sys.argv[1:])
