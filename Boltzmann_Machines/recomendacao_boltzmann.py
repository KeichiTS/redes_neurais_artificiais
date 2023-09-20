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

filmes = ['A bruxa', 'Invocação do Mal', 'O Chamado', 'Se beber não case', 'Gente Grande', 'American Pie']

rbm.train(base, max_epochs = 5000)
rbm.weights

usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])

rbm.run_visible(usuario1)
rbm.run_visible(usuario2)

camada_escondida = np.array([[1,0]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario1[0])):
    if usuario1[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])