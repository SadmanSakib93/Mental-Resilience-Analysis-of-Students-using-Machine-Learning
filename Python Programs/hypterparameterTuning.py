# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:06:00 2019

@author: Sadman Sakib
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

algoChoice=-1
csvHeaderName=''
outputFileName=''
yLabelText=''
#***CHANGE FILE NAME FOR DIFFERENT RESULTS ***
# 2 Class (Age categorized) file name = datasetOneHotEncoded_AgeCategorized_2.csv
# 2 Class (Age numeric) file name = datasetOneHotEncoded_AgeNumeric_2.csv
# 3 Class (Age categorized) file name = datasetOneHotEncoded_AgeCategorized_3.csv
# 3 Class (Age numeric) file name = datasetOneHotEncoded_AgeNumeric_3.csv
df = pd.read_csv("datasetOneHotEncoded_AgeNumeric_3.csv")
df.head()
# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:len(df.columns)-2]) 	  # end index is exclusive
y = np.array(df['Resilience level']) 

scaleColumns=[]
scaleColumnsNames=[]
columnNames=df.iloc[:, 0:len(df.columns)-2].columns
indx=0
for eachCol in columnNames:
    colMax=df[eachCol].max()
    if(colMax!=1):
        scaleColumns.append(indx)
        scaleColumnsNames.append(eachCol)
    indx+=1

#***NORMALIZATION***
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X[:,scaleColumns])
#****Z-SCORE****
scaler = StandardScaler()
X[:,scaleColumns] = scaler.fit_transform(X[:,scaleColumns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=66) 

print("1. KNN\n2. Random Forest\n3. MLP")
algoChoice=int(input("Enter Your Choice:")) 
if(algoChoice==1):
    #******KNN*********
    k_range = list(range(1, 31))
    param_grid = {'n_neighbors':k_range}
    clf = KNeighborsClassifier()
    csvHeaderName=['K']
    outputFileName='knn_hyperparameter_tuning'
    yLabelText='Value of K'

elif(algoChoice==2):
    #****** RR ********
    num_of_trees=list(range(2, 121))
    param_grid = {'n_estimators':num_of_trees}
    clf=RandomForestClassifier()
    csvHeaderName=['Number of Trees']
    outputFileName='rf_hyperparameter_tuning'
    yLabelText='Number of trees'
    
elif(algoChoice==3):
    #****** Neural Network ******
    param_grid = {
#        Please comment out one of the following two lines (to select hidden_layer_sizes) according to selected files
#        2 CLASS
#        'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),
#                              (11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,)],
#        3 CLASS
        'hidden_layer_sizes': [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3),(10,3),
                              (11,3), (12,3),(13,3),(14,3),(15,3),(16,3),(17,3),(18,3),(19,3),(20,3)],                               
#        'activation': ['tanh', 'relu'],
#        'alpha': [0.0001, 0.9],
    }
    clf = MLPClassifier()
    csvHeaderName=['Number of Nodes']
    outputFileName='mlp_hyperparameter_tuning'
    yLabelText='Number of nodes'
    
print("param_grid",param_grid)
## instantiate the grid
grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)

# create a list of the mean scores only
print("***")
cv_results=grid.cv_results_
accForAllParms=cv_results['mean_test_score']
print(accForAllParms)
# Single best score achieved across all params (k)
print("best_score_",grid.best_score_)
# Dictionary containing the parameters (k) used to generate that score
print("best_params_",grid.best_params_)
print("best_estimator_",grid.best_estimator_)
accForAllParmsCount=list(range(1,len(accForAllParms)+1))
df_accForAllParms=pd.DataFrame(accForAllParmsCount, columns = csvHeaderName)
df_accForAllParms['Accuracy']=accForAllParms
df_accForAllParms.to_csv('Outputs\\'+outputFileName+'.csv', index=False)
print("***Performance on Train data***")
print("Accuracy: %.2f %%" %(100*(grid.best_score_)))


#********PLOT GRAPH******
fig=plt.rc('figure', figsize=(12, 7))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 28}
fontLabels = {'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 26,
        }
fig = plt.figure()
y_pos = np.arange(1,len(accForAllParmsCount)+1)
plt.bar(y_pos, accForAllParms, color=['blue'], alpha=0.5)
plt.xticks(fontsize=18)
plt.ylim(ymin=(accForAllParms.max()/2))  
plt.yticks(fontsize=18)
plt.ylabel("Accuracy", fontdict=fontLabels)
plt.xlabel(yLabelText, fontdict=fontLabels)
plt.rc('font', **font)
plt.savefig('Outputs\\'+outputFileName+'.png', dpi=900)
plt.show()


#*******TESTING********
#APPLY BEST PARAMS
clf = grid.best_estimator_
# fitting the model
clf.fit(X_train, y_train)
# predict the response
y_predict = clf.predict(X_test)
# evaluate accuracy
print("***Performance on Test data***")
print("Best accuracy for, ",grid.best_params_)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_predict)))
print("Precision score:",precision_score(y_test, y_predict, average="macro"))