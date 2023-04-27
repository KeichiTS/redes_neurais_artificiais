# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:45:12 2023

@author: KeichiTS
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')