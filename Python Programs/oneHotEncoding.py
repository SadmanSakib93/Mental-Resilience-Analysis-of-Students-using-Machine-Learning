# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:55:57 2019

@author: Sadman Sakib
"""

# import libraries 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import csv
# import the data required 
data = pd.read_csv("datasetOneHotEncoded_AgeCategorized.csv") 
print(data)
columnHeaders=['Age0', 'Age1', 'Age2', 'Clerkship','Clinical Sciences','Basic Sciences',
               'OverallQOL_2', 'OverallQOL_3', 'OverallQOL_4', 'OverallQOL_5', 'OverallQOL_6', 'OverallQOL_7', 'OverallQOL_8', 'OverallQOL_9', 'OverallQOL_10',
               'MedicalQOL_0', 'MedicalQOL_1', 'MedicalQOL_2', 'MedicalQOL_3', 'MedicalQOL_4', 'MedicalQOL_5', 'MedicalQOL_6', 'MedicalQOL_7', 'MedicalQOL_8', 'MedicalQOL_9', 'MedicalQOL_10',
               'Sex', 'WHOQOL_physical_health', 'WHOQOL_psychological', 'WHOQOL_social_relationships', 'WHOQOL_environment', 'DREEM_learning', 'DREEM_teachers', 'DREEM_academic_self_perception',
               'DREEM_atmosphere', 'DREEM_social_self_perception', 'BDI',  'School_legal_status', 'School_location', 'State_Anxiety', 'Trait_anxiety', 'Resilience score', 'Resilience level']
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",       
         OneHotEncoder(), 
         [34]            
         )
    ],
    remainder='passthrough' 
)
dataAfter = transformer.fit_transform(data)
dfEncoded = pd.DataFrame(dataAfter,columns=columnHeaders)
dfEncoded.to_csv('datasetOneHotEncoded_AgeCategorized.csv', encoding='utf-8', index=False)