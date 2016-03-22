import scipy.io as spio
import sys
import csv
import pickle

def main():
       
    dict = {}
    for x in range(1000):
        dict[x] = []

    with open('mfcc_tsne.csv') as f:
        reader = csv.reader(f)  
        for row in reader:
            #print len(row)
            i = 0
            for x in row:
                dict[i].append(float(x))
                i += 1
        print len(dict[0])
        pickle.dump(dict, open('mfcc_tsne.p', 'wb'))

           
    
if __name__ == "__main__":
    main()
