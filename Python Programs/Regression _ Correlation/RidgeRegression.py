# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:34:05 2019

@author: Tazrin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

data = pd.read_csv("datasetOneHotEncoded_AgeNumeric.csv")
x = data.iloc[:, :-2].values
y = data.iloc[:, -2].values

#scaling the features
scaling = MinMaxScaler(feature_range=(0,1)).fit(x)
x = scaling.transform(x)

#scaling y by dividing by 100
y = y/100


clf = Ridge(alpha=1.0)
Cv = KFold(n_splits=5, shuffle = True , random_state=10)
y_pred = cross_val_predict(clf, x, y, cv=Cv)

#calculating the error
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))