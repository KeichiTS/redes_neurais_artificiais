# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:26:14 2023

@author: KeichiTS
"""

from minisom import MiniSom
import pandas as pd
import numpy as np

base = pd.read_csv('personagens.csv')

base.loc[base.classe == 'Bart', 'classe'] = 0
base.loc[base.classe == 'Homer', 'classe'] = 1

X = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(x = 9, y = 9, input_len = 6, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot 
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor = 'None', markersize = 10,
         markeredgewidth = 2, markeredgecolor = colors[y[i]],)
    
mapeamento = som.win_map(X)
suspeitos = mapeamento[(4,3)]
suspeitos = normalizador.inverse_transform(suspeitos)