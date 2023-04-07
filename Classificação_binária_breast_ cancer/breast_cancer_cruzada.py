# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:34:07 2023

@author: KeichiTS
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer = 'random_uniform', input_dim = 30))
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    optimizador = keras.optimizers.Adam( lr = 0.001, decay = 0.0001, clipvalue = 0.5 )
    classificador.compile(optimizer = optimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 180,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()