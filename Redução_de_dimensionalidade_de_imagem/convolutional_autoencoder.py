# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:44:46 2023

@author: KeichiTS
"""

import matplotlib.pyplot as plt 
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()
previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), 28 , 28 , 1)
previsores_teste = previsores_teste.reshape(len(previsores_teste), 28 , 28 , 1)


previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu',
                       input_shape = (28,28,1)))
autoencoder.add(MaxPooling2D(pool_size = (2,2)))

autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding = 'same'))
autoencoder.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))

#4, 4, 8
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding = 'same', strides = (2,2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape(4, 4, 8))

