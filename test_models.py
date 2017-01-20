from feature_format import feature_engineering
from imblearn.over_sampling import SMOTE
import json
import numpy as np
import pandas as pd
import pickle
from Classifiers import Classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

def dump_classifier(clf, filename):
    with open(filename, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)

def cross_validate(clf, X, y):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    precisions = []
    recalls = []
    f1s = []
    print("\n____________{}____________".format(clf.__class__.__name__))
    for train_index, test_index in sss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ## SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_sample(X_train, y_train)
        ### fit the classifier using training set, and test on validation set
        clf.fit(X_res, y_res)
        predictions = clf.predict(X_test)
        precisions.append(precision_score(y_test, predictions))
        recalls.append(recall_score(y_test, predictions))
        f1s.append(f1_score(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
    print("Recall: {:.3%}".format(np.mean(recalls)))
    print("Precision: {:.3%}".format(np.mean(precisions)))
    print("F1: {:.3%}".format(np.mean(f1s)))


if __name__ == '__main__':
    ## get training features
    with open('data/X_train.json') as f:
        data = json.load(f)
    X, feature_names = feature_engineering(data)
    import pdb; pdb.set_trace()

    ## get training labels
    with open('data/y_train.json') as f:
        labels = json.load(f)
    df_y = pd.DataFrame(labels)
    # get y
    y = df_y.pop('fraud').values

    '''INITIAL MODEL SELECTION
    models = [LogisticRegression(), RandomForestClassifier(),
        AdaBoostClassifier(), GradientBoostingClassifier()]
    for model in tuned_models:
        cross_validate(model, X, y)
    '''

    '''HYPERPARAMETER TUNING
    rf_grid = {"n_estimators":[10,50,100,150,200],
                "min_samples_split":[2, 4, 6, 8, 10]}
    gb_grid = {"learning_rate":[0.001, 0.01, 0.1, 0.2, 0.5],
                "n_estimators":[100, 150, 200],
                "max_depth":[2, 3, 4, 5]}
    rf_clf = GridSearchCV(RandomForestClassifier(), rf_grid, scoring='recall', cv=5)
    rf_clf.fit(X,y)
    rf_clf.best_estimator_
    gb_clf = GridSearchCV(GradientBoostingClassifier(), gb_grid, scoring='recall', cv=5)
    gb_clf.fit(X,y)
    gb_clf.best_estimator_
    '''
    tuned_models = [RandomForestClassifier(n_estimators=150),
                    GradientBoostingClassifier(learning_rate=0.02, max_depth=3,
                                                n_estimators=200)]
    clf = Classifiers(tuned_models)
    clf.cross_validate(X, y)
    clf.train(X, y)
    '''
    clf.plot_roc_curve()
    # # confusion matrix = [[TN, FP],[FN, TP]]
    # # FN = refund cost of ticket and also probably lose the customer ~$100
    # # FP = contact the seller, do some verfication ~$20
    cb = np.array([[0, -20],[-100, -20]])
    clf.plot_profit(cb)
    '''

    '''FEATURE IMPORTANCES'''
    print(feature_names[np.argsort(clf.classifiers[1].feature_importances_)])

    '''FINAL MODEL CREATION'''
    final_model = RandomForestClassifier(n_estimators=150)
    # sm = SMOTE(random_state=42)
    # X_smote, y_smote = sm.fit_sample(X, y)
    # final_model.fit(X_smote, y_smote)
    final_model.fit(X,y)
    dump_classifier(final_model, "our_classifier.pkl")
