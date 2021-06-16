# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:05:57 2019

@author: Sadman Sakib
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def generateResilienceLevels(fileName, kValue):
    # loading training data
    df = pd.read_csv(fileName+".csv")
    df.head()
    # create design matrix X and target vector y
    X = np.array(df['Resilience score'])  
    X=X.reshape(-1,1)
    xrange=range(0,len(X))
    
    
    kmeans = KMeans(n_clusters=kValue, random_state=123).fit(X)
    cluster_labels=kmeans.labels_
    centers=kmeans.cluster_centers_
    print(cluster_labels)
    print(centers)
    print(len(cluster_labels))
    plt.figure(figsize=(10, 7))
    plt.scatter(xrange,X ,c=cluster_labels,cmap='brg')
    plt.title("K-means Clustering (k="+str(kValue)+")")
    plt.xlabel("Instances")
    plt.ylabel("Resilience Score")
    plt.show()
    
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", kValue,
          "The average silhouette_score is :", silhouette_avg)
    
    if(kValue==3):
        cluster_labels_updated=[]
        for valIndex in cluster_labels:
            if(valIndex==0):
                cluster_labels_updated.append(1)
            elif(valIndex==1):
                cluster_labels_updated.append(2)
            elif(valIndex==2):
                cluster_labels_updated.append(0)
        cluster_labels_updated=np.asarray(cluster_labels_updated)  
        
    cluster_labels=cluster_labels_updated     
    df['Resilience level']=cluster_labels_updated
    df.to_csv(fileName+'_'+str(kValue)+'.csv', encoding='utf-8', index=False)



kValue=int(input("Enter the number of resilience levels to generate (K of K-Means clusterings):"))
generateResilienceLevels("datasetOneHotEncoded_AgeCategorized",kValue)
generateResilienceLevels("datasetOneHotEncoded_AgeNumeric",kValue)
