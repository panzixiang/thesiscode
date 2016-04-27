import pickle
import sys


def main():
    arxiv_11 = pickle.load(open("2011_big_pop.p", "rb"))
    filename = sys.argv[1]
    doc_set = arxiv_11[filename]

    for article in doc_set:
        print article
        print '\t'

if __name__ == "__main__":
    main()             