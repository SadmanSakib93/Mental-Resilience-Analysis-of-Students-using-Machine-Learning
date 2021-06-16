# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:36:23 2019

@author: Sadman Sakib
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTENC
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# loading training data
df = pd.read_csv("datasetOneHotEncoded_AgeNumeric.csv")

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:len(df.columns)-2]) 	  # end index is exclusive
Y = np.array(df['Resilience level']) 	  # another way of indexing a pandas df

#MIN-MAX
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
#STANDARD SCALAR
scaler = StandardScaler()
X = scaler.fit_transform(X)

allAccuracy=[]
allPrecision=[]
def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    recall=recall_score(y_true, y_pred,average='macro')
    allAccuracy.append(accuracy)
    allPrecision.append(precision)
#    print("Accuracy  : {}".format(accuracy))
#    print("Precision : {}".format(precision))
#    print("Recall    : {}".format(recall))
#    print("Confusion Matrix:")
#    cm=confusion_matrix(y_true, y_pred)
#    print(cm)
    return accuracy, precision, recall



for itr in range(3):
    print(itr)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, Y)
    StratifiedKFold(n_splits=10, random_state=122, shuffle=True)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    #    print("BEFORE",len(X_train))
        sm = SMOTENC(random_state=123, categorical_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 36])
        X_train, Y_train = sm.fit_resample(X_train, Y_train)
    #    print("AFTER",len(X_train))
        if(itr==0):
            clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=None,n_neighbors=5, p=2,
                                 weights='uniform')
            print("Calculating performance of KNN . . .")
        elif(itr==1):
            clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=73,
                               n_jobs=None, oob_score=False, random_state=None,
                               verbose=0, warm_start=False)
            print("Calculating performance of RF . . .")
        elif(itr==2):
            clf=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                          beta_2=0.999, early_stopping=False, epsilon=1e-08,
                          hidden_layer_sizes=(15,), learning_rate='constant',
                          learning_rate_init=0.001, max_iter=200, momentum=0.9,
                          n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                          random_state=None, shuffle=True, solver='adam', tol=0.0001,
                          validation_fraction=0.1, verbose=False, warm_start=False)
            print("Calculating performance of MLP . . .")
        
        clf.fit(X_train, Y_train)
        y_predict = clf.predict(X_test)
        my_metrics(Y_test, y_predict)
    
    print("Mean Accuracy:",np.mean(allAccuracy))
    print("Mean Precision:",np.mean(allPrecision))
    allAccuracy.clear()
    allPrecision.clear()
