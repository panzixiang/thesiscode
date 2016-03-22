#!/bin/bash

python trial.py 'knn3' | python createCsv.py knn3
python trial.py 'knn5' | python createCsv.py knn5
python trial.py 'gnb' | python createCsv.py gnb
python trial.py 'lda' | python createCsv.py lda 
python trial.py 'qda' | python createCsv.py qda
python trial.py 'svmLin' | python createCsv.py svmLin
python trial.py 'svmRbf' | python createCsv.py svmRbf
python trial.py 'svmSig' | python createCsv.py svmSig