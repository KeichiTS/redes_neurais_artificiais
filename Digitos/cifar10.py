# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:16:41 2023

@author: KeichiTS
"""

import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization


(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

plt.imshow(X_treinamento[10])

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste,10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (32,32,3), 
                         activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
#classificador.add(Flatten())

classificador.add(Conv2D(32,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())


classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', 
                      optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 5, 
                  validation_data=(previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste,classe_teste)