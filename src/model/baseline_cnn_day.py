#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten, Reshape
from keras.layers import add, concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras import backend as K

def build_model(time_dim=92):
    x_t_c = Input(shape=(time_dim, 34), dtype='float32', name='day/x_t_c')
    x_c_c = Input(shape=(2,), dtype='float32', name='x_c_c')
    x_c_d_1 = Input(shape=(1,), dtype='int32', name='x_c_d_1')
    x_c_d_2 = Input(shape=(1,), dtype='int32', name='x_c_d_2')
#    x_wide = Input(shape=(122,), dtype='float32', name='x_wide')

    embedding_2_d_1 = Embedding(input_dim=4+2, output_dim=5, input_length=1, name='embedding_2_d_1')(x_c_d_1)
    embedding_2_d_2 = Embedding(input_dim=2+2, output_dim=5, input_length=1, name='embedding_2_d_2')(x_c_d_2)
    embedding_2_d_1 = Flatten(name='flatten_2_d_1')(embedding_2_d_1)
    embedding_2_d_2 = Flatten(name='flatten_2_d_2')(embedding_2_d_2)

    x = x_t_c
    x = BatchNormalization(name='bn_input')(x)

    x = Conv1D(250, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)

    x = concatenate([x, x_c_c, embedding_2_d_1, embedding_2_d_2])
    x = Dense(250)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model= Model(inputs=[x_t_c, x_c_c, x_c_d_1, x_c_d_2], outputs=output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()

