# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:30:26 2023

@author: KeichiTS
"""

from minisom import MiniSom
import pandas as pd

base = pd.read_csv('wines.csv')

X = base.iloc[:,1:14].values
y = base.iloc[:,0].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))

X = normalizador.fit_transform(X)

som = MiniSom( x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2 )
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

som._weights
som._activation_map
q = som.activation_response(X)

from pylab import pcolor, colorbar
pcolor(som.distance_map().T)
#MID - mean inter neuron distance 
colorbar()