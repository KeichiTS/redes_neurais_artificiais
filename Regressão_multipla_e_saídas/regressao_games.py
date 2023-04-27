# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:45:12 2023

@author: KeichiTS
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
#plataforma_jogos = base.Platform  #apenas um teste para ver como funcionada o base
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:,4].values
venda_eu = base.iloc[:,5].values
venda_jp = base.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])

#onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8]) #essa linha precisou de ajustes 
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()