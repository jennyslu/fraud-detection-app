from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc

class Classifiers(object):
    '''
    Classifier object for fitting, storing, and comparing multiple model outputs.
    '''
    def __init__(self, classifier_list):
        self.classifiers = classifier_list
        self.classifier_names = [est.__class__.__name__ for est in self.classifiers]

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.25, random_state=42)
        for clf in self.classifiers:
            clf.fit(X_train, y_train)
        self._X_test = X_test
        self._y_test = y_test

    def cross_validate(self, X, y):
        for clf in self.classifiers:
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


    def plot_roc_curve(self):
        fig, ax = plt.subplots()
        for name, clf in zip(self.classifier_names, self.classifiers):
            predict_probas = clf.predict_proba(self._X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(self._y_test, predict_probas, pos_label=1)
            roc_auc = auc(x=fpr, y=tpr)
            ax.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc))
        # 45 degree line
        x_diag = np.linspace(0, 1.0, 20)
        ax.plot(x_diag, x_diag, color='grey', ls='--')
        ax.legend(loc='best')
        ax.set_ylabel('True Positive Rate', size = 20)
        ax.set_xlabel('False Positive Rate', size = 20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        fig.set_size_inches(15, 10)
        fig.savefig('ROC_curves.png', dpi=100)


    def plot_profit(self, cb):
        fig, ax = plt.subplots()
        percentages = np.linspace(0, 100, len(self._y_test) + 1)
        for name, clf in zip(self.classifier_names, self.classifiers):
            probabilities = clf.predict_proba(self._X_test)[:,1]
            thresholds = sorted(probabilities)
            thresholds.append(1.0)
            profits = []
            for threshold in thresholds:
                y_predict = probabilities >= threshold
                confusion_mat = confusion_matrix(self._y_test, y_predict)
                profit = np.sum(confusion_mat * cb) / float(len(self._y_test))
                profits.append(profit)
            ax.plot(percentages, profits, label = name)
        ax.legend(loc='best')
        ax.set_ylabel('Profit', size = 20)
        ax.set_xlabel('Proportion predicted fraud', size = 20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        fig.set_size_inches(15, 10)
        fig.savefig('profit_curves.png')
