# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:38:34 2023

@author: KeichiTS
"""

import minisom as MiniSom
import pandas as pd 
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:,0:4].values
y = base.iloc[:,4].values

from sklearn.preprocessing import MinMaxScaler
normalizador= MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)