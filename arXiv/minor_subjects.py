import pickle
import sys

topicDict = {}
topicDict['astro'] = 6
topicDict['cond'] = 8
topicDict['cs'] = 40
topicDict['hep'] = 4
topicDict['math'] = 31
topicDict['physics'] = 28
topicDict['qbio'] = 10
topicDict['qfin'] = 9
topicDict['quant'] = 28
topicDict['stat'] = 2

pickle.dump(bigcatDict, open("minor_subjects.p", "wb"))   