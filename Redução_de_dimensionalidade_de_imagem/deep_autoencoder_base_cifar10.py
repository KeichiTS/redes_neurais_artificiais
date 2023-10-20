# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:22:35 2023

@author: KeichiTS
"""

import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = cifar10.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape(len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape(len(previsores_teste), np.prod(previsores_teste.shape[1:]))


# 3072 - 128 - 64 - 128 - 3072
autoencoder = Sequential()

#encode
autoencoder.add(Dense(units = 128, activation = 'relu', input_dim = 3072))
autoencoder.add(Dense(units = 64, activation = 'relu'))

#Decode
autoencoder.add(Dense(units = 128, activation = 'relu'))
autoencoder.add(Dense(units = 3072, activation = 'sigmoid'))

autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256, 
                validation_data = (previsores_teste, previsores_teste))

dimensao_original = Input(shape = (3072,))
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]
encoder = Model(dimensao_original, 
                camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))
encoder.summary()

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)
plt.figure(figsize = (32,32))
for i, indice_imagem in enumerate(imagens_teste):
    #print(i)
    #print(indice_imagem)
    
    # imagem original
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16,8))
    plt.xticks(())
    plt.yticks(())
    
    # imagem reconstruida
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2 )
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())