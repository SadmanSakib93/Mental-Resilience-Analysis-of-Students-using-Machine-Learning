# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:05:57 2019

@author: Sadman Sakib
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# loading training data
df = pd.read_csv("datasetOneHotEncoded_AgeCategorized.csv")
df.head()
k=-1
columns=['Average Silhouette Score']
# create design matrix X and target vector y
X = np.array(df['Resilience score'])  	  # end index is exclusive
#y = np.array(df['Resilience level']) 	  # another way of indexing a pandas df
X=X.reshape(-1,1)
xrange=range(0,len(X))
all_silhouette_score=[]
fontLabels = {'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 22,
        }
for clusterNum in range(2,11):
    k=clusterNum
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    cluster_labels=kmeans.labels_

    fig = plt.figure(figsize=(13, 8))
    plt.scatter(xrange,X ,c=cluster_labels,cmap='brg')
    plt.title("K-means Clustering (k="+str(k)+")", fontdict=fontLabels)
    plt.xlabel("Instances", fontdict=fontLabels)
    plt.ylabel("Resilience Score", fontdict=fontLabels)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
#    
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)
    all_silhouette_score.append(silhouette_avg)
    
    # Compute the silhouette scores for each sample
#    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
np.savetxt("Outputs\kmeans_silhouette_score.csv", all_silhouette_score, delimiter=",")
bestClusterK=np.argmax(all_silhouette_score)+2

kmeans = KMeans(n_clusters=bestClusterK, random_state=0).fit(X)
cluster_labels=kmeans.labels_
centers=kmeans.cluster_centers_
print(cluster_labels)
print(centers)

fig = plt.figure(figsize=(13, 8))
plt.scatter(xrange,X ,c=cluster_labels,cmap='brg')
plt.title("K-means Clustering (k="+str(bestClusterK)+")", fontdict=fontLabels)
plt.xlabel("Instances", fontdict=fontLabels)
plt.ylabel("Resilience Score", fontdict=fontLabels)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", bestClusterK,
      "The average silhouette_score is :", silhouette_avg)

df['Resilience level']=cluster_labels
