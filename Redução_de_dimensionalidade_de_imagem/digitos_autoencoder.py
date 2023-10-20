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