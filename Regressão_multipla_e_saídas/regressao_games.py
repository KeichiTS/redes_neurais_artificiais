# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:45:12 2023

@author: KeichiTS
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
#plataforma_jogos = base.Platform  #apenas um teste para ver como funcionada o base
base = base.drop('Name', axis = 1)

