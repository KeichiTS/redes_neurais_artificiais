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