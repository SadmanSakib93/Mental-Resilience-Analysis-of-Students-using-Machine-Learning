# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:14:27 2019

@author: Tazrin
"""

import numpy as np
import pandas as pd
from scipy import stats
data = pd.read_csv("dataset.csv")
a = data.iloc[:, -2].values
corr = []
for column in data:
    b = data[column].values
    c = stats.spearmanr(a,b).correlation
    corr.append(c)


#print correlations sorted in descending order
column = data.columns
cor = sorted(zip(corr, column), reverse=True)

for i in cor:
    print(i)
