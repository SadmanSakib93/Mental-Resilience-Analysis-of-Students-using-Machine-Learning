# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:48:24 2019

@author: Sadman Sakib
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# loading training data
df = pd.read_csv("datasetOneHotEncoded_AgeNumeric.csv")

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:len(df.columns)-2]) 	
Y = np.array(df['Resilience score']) 	 

bin_labels = [0,1,2]
print(pd.qcut(df['Age'], q=3, labels=bin_labels).value_counts())
print(pd.qcut(df['Age'], q=3).value_counts())
dfAfter=df.copy()
dfAfter['Age']=pd.qcut(df['Age'], q=3, labels=bin_labels)
dfAfter.to_csv('datasetOneHotEncoded_AgeCategorized.csv', encoding='utf-8', index=False)