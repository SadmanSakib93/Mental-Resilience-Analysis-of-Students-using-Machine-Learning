# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:03:28 2019

@author: Sadman Sakib
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def makeDictFeatureImportance(features, importance):
    featureDic=dict()
    for indx in range(len(features)):
        featureDic[features[indx]]=importance[indx]
#    print("featureDic",featureDic)    
    return featureDic

# loading training data
df = pd.read_csv("dataset.csv")
df.head()

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:len(df.columns)-2]) 	  # end index is exclusive
y = np.array(df['Resilience level']) 

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #

featureNames=df.iloc[:, 0:len(df.columns)-2].columns
#featureNames=list(featureNames[2:])
	

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

featureImportance=clf.feature_importances_
featureImpDict=makeDictFeatureImportance(featureNames,featureImportance)
#featureImpDictSorted=sorted(featureImpDict.items(), key=lambda x: x[1], reverse=True)
#print(dict(featureImpDictSorted))
#featureImpFromDictSorted2_DF=pd.DataFrame.from_dict(featureImpDictSorted, orient='index')
importanceCorrelationList=[list(featureNames),list(featureImportance),list(range(19))]

corr_matrix = df.corr()
corr_with_resilience=corr_matrix["Resilience score"].sort_values(ascending=False)
corr_index_names=corr_with_resilience.index.tolist()
#print(corr_matrix["Resilience score"].sort_values(ascending=False))

#**** FEATURE IMPORTANCE + CORRELATION ****
for eachNameIndex in range(2,len(corr_index_names)):
    corr_val=corr_with_resilience.get(key = corr_index_names[eachNameIndex])
    print(corr_index_names[eachNameIndex],corr_val)
    addIndex=importanceCorrelationList[0].index(corr_index_names[eachNameIndex])
    importanceCorrelationList[2][addIndex]=corr_val

importanceCorrelationArray=np.asarray(importanceCorrelationList).T

saveDF=pd.DataFrame(importanceCorrelationArray, columns=['Feature', 'Feature Importance', 'Correlation with Resilience'])
saveDF=saveDF.sort_values(by=['Feature Importance'], ascending=False)
saveDF.to_csv("Outputs\FeatureImportance_Correlation.csv", index=False)