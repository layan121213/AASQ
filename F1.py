from sklearn.metrics import f1_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

#labeling
def label(groundTruth):
    labels = ["not","semi","sim"]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    le.transform(labels)
    actual = le.inverse_transform(groundTruth)
    return actual

#f1 score & precision & recall
def evaluationMetric(actual,predicted):
    predicted= [0 if (i <= 0.41) else 1 if (0.41 <= i <= 0.60) else 2 if (0.60 <= i <= 1) else i for i in predicted] 
    labeledActual = label(actual)
    labeledPredicted = label(predicted)
    precision, recall, fscore, support = score(labeledActual, labeledPredicted, average='weighted')
    return ('fscore: {}%'.format(int(fscore*100))) 

