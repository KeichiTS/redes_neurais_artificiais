# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:30:26 2023

@author: KeichiTS
"""

from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

# Load the data
classe = pd.read_csv('entradas_breast.csv')
previsores = pd.read_csv('saidas_breast.csv')

X = classe.values
y = previsores.values

# Normalize the data
normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)

# Initialize and train the SOM
som = MiniSom(x = 11, y = 11, input_len = 30, sigma = 3, learning_rate = .5 , random_seed = 0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration= 1000)

# Get the coordinates of the winning neurons for each input
winning_neurons = np.array([som.winner(x) for x in X])

# Define markers and colors for plotting
markers = ['o', 's']
colors = ['r', 'g']

# Plot the SOM
pcolor(som.distance_map().T)
colorbar()

# Plot the markers for each input
for i, x in enumerate(X):
    w = winning_neurons[i]
    marker_index = int(y[i].item())  # Convert to int
    plot(w[0] + 0.5, w[1] + 0.5, markers[marker_index],
         markerfacecolor='None', markersize=10,
         markeredgecolor=colors[marker_index], markeredgewidth=2)