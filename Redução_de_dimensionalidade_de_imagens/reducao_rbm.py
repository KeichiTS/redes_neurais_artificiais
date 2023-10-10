# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:25:59 2023

@author: KeichiTS
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
previsores = np.array(base.data, 'float32')
classe = base.target

normalizador = MinMaxScaler(feature_range = (0,1))
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,
                                                                                              classe,
                                                                                              test_size = 0.2,
                                                                                              random_state = 0)