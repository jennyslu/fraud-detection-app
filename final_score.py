from feature_format import feature_engineering
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


def load_classifier(filename):
    with open(filename, "rb") as clf_infile:
        clf = pickle.load(clf_infile)
    return clf

if __name__ == '__main__':
    ## get training features
    with open('data/X_test.json') as f:
        data = json.load(f)
    X, feature_names = feature_engineering(data)

    ## get training labels
    with open('data/y_test.json') as f:
        labels = json.load(f)
    df_y = pd.DataFrame(labels)
    # get y
    y = df_y.pop('fraud').values

    clf = load_classifier("our_classifier.pkl")
    predictions = clf.predict(X)
    print(confusion_matrix(y, predictions))
    print("Recall: {:.3%}".format(precision_score(y, predictions)))
    print("Precision: {:.3%}".format(recall_score(y, predictions)))
    print("F1: {:.3%}".format(f1_score(y, predictions)))

    '''FINAL SCORES
    [[3236   21]
     [  39  289]]
    Recall: 93.226%
    Precision: 88.110%
    F1: 90.596%
    '''
