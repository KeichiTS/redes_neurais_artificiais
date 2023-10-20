# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:59:33 2023

@author: KeichiTS
"""

import matplotlib.pyplot as plt
import numpy as np 
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense 

(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))

# redução de dimensionalidade de 784 para 32 

fator_compactacao = 784 / 32

autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.summary()
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))