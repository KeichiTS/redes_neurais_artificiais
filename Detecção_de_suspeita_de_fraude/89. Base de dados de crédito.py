# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:38:34 2023

@author: KeichiTS
"""

from minisom import MiniSom
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0)
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