# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:13:53 2023

@author: KeichiTS
"""

from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 2)

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])
rbm.train(base, max_epochs = 5000)
rbm.weights