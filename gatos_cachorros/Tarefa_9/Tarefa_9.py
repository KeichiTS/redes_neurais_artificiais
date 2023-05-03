# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:10:38 2023

@author: KeichiTS
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import numpy as np
#from keras.preprocessing import image
from keras.utils import load_img, img_to_array

classificador = Sequential()
classificador.add(Conv2D(32,(3,3), input_shape = (64,64,3), 
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32,(3,3), input_shape = (64,64,3), 
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 4, activation = 'relu'))
#classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'relu'))
#classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, 
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2
                                         )
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset_personagens/training_set',
                                                           target_size = (64,64),
                                                           batch_size = 10,
                                                           class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('dataset_personagens/test_set',
                                               target_size = (64, 64),
                                               batch_size = 10,
                                               class_mode = 'binary')

classificador.fit_generator(base_treinamento, steps_per_epoch = 20, 
                            epochs = 100, validation_data = base_teste,
                            validation_steps = 72
                            )

imagem_teste = load_img('dataset_personagens/test_set/bart/bart2.bmp',
                              target_size = (64,64))
imagem_teste = img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)

base_treinamento.class_indices