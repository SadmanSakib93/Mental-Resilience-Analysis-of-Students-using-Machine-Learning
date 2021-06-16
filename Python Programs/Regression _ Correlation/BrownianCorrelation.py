# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:17:35 2019

@author: Tazrin
"""

from scipy.spatial.distance import pdist, squareform
import numpy as np
#Brownian Correlation Function
def browniancorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

import pandas as pd
data = pd.read_csv("dataset.csv")
a = data.iloc[:, -2].values
print("Brownian Correlation of each feature to resilience:")
corr = []
for column in data:
    b = data[column].values 
    corr.append(browniancorr(a, b))

#print correlations sorted in ascending order
column = data.columns
cor = sorted(zip(corr, column), reverse=True)

for i in cor:
    print(i)
