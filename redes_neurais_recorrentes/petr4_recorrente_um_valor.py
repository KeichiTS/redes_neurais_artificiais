# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:18:11 2023

@author: KeichiTS
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as pd 
import pandas as pd

base = pd.read_csv('petr4_treinamento.csv')