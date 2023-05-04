# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:18:11 2023

@author: KeichiTS
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as pd 
import pandas as pd

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:,1:2].values

normalizador = MinMaxScaler(feature_range = (0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)