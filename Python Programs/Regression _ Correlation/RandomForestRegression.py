# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:28:11 2019

@author: Tazrin
"""
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("datasetOneHotEncoded_AgeNumeric.csv")
x = data.iloc[:, :-2].values
y = data.iloc[:, -2].values

#scaling the features
scaling = MinMaxScaler(feature_range=(0,1)).fit(x)
x = scaling.transform(x)

#scaling y by dividing by 100
y = y/100
y=y.reshape(-1, 1)


#y = scaling.transform(y)


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
Cv = KFold(n_splits=5, shuffle = True , random_state=10)
y_pred = cross_val_predict(rf, x, y, cv=Cv)

#calculating the error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
