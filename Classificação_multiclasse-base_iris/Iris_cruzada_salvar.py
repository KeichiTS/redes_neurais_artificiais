# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:37:56 2023

@author: KeichiTS
"""

import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dropout(0.3))
#    classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation= 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
                      metrics = 'accuracy')

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file: 
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')