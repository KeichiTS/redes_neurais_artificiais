# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:02:52 2023

@author: KeichiTS
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)
base = base.drop('Name', axis = 1)

base = base.dropna(axis = 0)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
venda_global = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

ativacao = Activation(activation = 'relu')

camada_entrada = Input(shape=(303,))
camada_oculta1 = Dense(units = 152, activation = ativacao)(camada_entrada)
camada_oculta2 = Dense(units = 152, activation = ativacao)(camada_oculta1)
camada_saida = Dense(units =1, activation = 'linear')(camada_oculta1)

regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida])
regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(previsores, venda_global,
              epochs = 5000, batch_size = 100)

previsao = regressor.predict(previsores)
