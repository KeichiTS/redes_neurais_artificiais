# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:36:52 2023

@author: KeichiTS
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('poluicao.csv')
base = base.dropna()
base_treinamento = base.drop(columns = 'cbwd')
base_treinamento = base_treinamento.iloc[:,5:].values
