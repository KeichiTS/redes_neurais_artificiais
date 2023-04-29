# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:21:08 2023

@author: KeichiTS
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

#Lê o CSV com a base de dados 
base = pd.read_csv('personagens.csv')

#Separa os previsores da classe de acordo com os valores na tabela 
previsores = base.iloc[:,0:6].values
classe = base.iloc[:,6].values

#Transforma os valores de classe de string para int32
classe = LabelEncoder().fit_transform(classe)

#Separa valores para teste e treinamento, sendo os elas os previsores e classes. O teste foi setado para 25% dos dados enquanto o treinamento
#utilizará 75%. 
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)


#Cria a rede neural com 6 unidades de entrada e duas camadas ocultas com 4 neuronios cada. Os métodos estão setados para o treinamento de uma rede 
#binária que devolve apenas um valor na ultima camada. 
#O classificador foi setado para um batch_size de 10 dados para 100 épocas

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 6))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = 'binary_accuracy')
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 10000)

#resultados - demonstra o loss_function e o percentual de acerto da base de dados de teste
resultado = classificador.evaluate(previsores_teste,classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

#Resultados - Método de avaliação para ver o número de acertos e erros para cada personagem 
matriz_de_confusao = confusion_matrix(previsoes, classe_teste)
