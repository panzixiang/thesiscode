import pickle
import sys


def main():
    arxiv_11 = pickle.load(open("2011_big_pop.p", "rb"))
    # filename = sys.argv[1]
    # doc_set = arxiv_11[filename]
    doc_set = arxiv_11['astro'] + arxiv_11['cond'] + \
            arxiv_11['cs'] + arxiv_11['hep'] + \
            arxiv_11['math'] + arxiv_11['physics'] + \
            arxiv_11['qbio'] + arxiv_11['qfin'] + \
            arxiv_11['quant'] + arxiv_11['stat'] 

    for i in range(len(doc_set)):
        print str(i) + '\t' + doc_set[i]

if __name__ == "__main__":
    main()             