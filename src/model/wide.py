#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import add, multiply, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam

def GatedCNN(input_tensor, nb_filter, kernel_size, stage, block):
    conv_name_base = 'conv' + str(stage) + block + '-'
    A = Conv1D(nb_filter, kernel_size, padding='same', name=conv_name_base+'a')(input_tensor)
    B = Conv1D(nb_filter, kernel_size, padding='same', name=conv_name_base+'b')(input_tensor)
    B_sig = Activation('sigmoid')(B)
    x = multiply([A, B_sig])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(time_dim=12):
    x_wide = Input(shape=(122,), dtype='float32', name='x_wide')

    output = Dense(1, activation='sigmoid')(x_wide)

    model= Model(inputs=x_wide, outputs=output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()

