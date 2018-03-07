#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import add, multiply, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
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
    x_t_c = Input(shape=(time_dim, 34), dtype='float32', name='day/x_t_c')
    x_c_c = Input(shape=(2,), dtype='float32', name='x_c_c')
    x_c_d_1 = Input(shape=(1,), dtype='int32', name='x_c_d_1')
    x_c_d_2 = Input(shape=(1,), dtype='int32', name='x_c_d_2')
    x_wide = Input(shape=(122,), dtype='float32', name='x_wide')

    embedding_2_d_1 = Embedding(input_dim=4+2, output_dim=5, input_length=1, name='embedding_2_d_1')(x_c_d_1)
    embedding_2_d_2 = Embedding(input_dim=2+2, output_dim=5, input_length=1, name='embedding_2_d_2')(x_c_d_2)
    embedding_2_d_1 = Flatten(name='flatten_2_d_1')(embedding_2_d_1)
    embedding_2_d_2 = Flatten(name='flatten_2_d_2')(embedding_2_d_2)

    x = x_t_c
    x = BatchNormalization(name='bn_input')(x)
    x = Conv1D(64, 1, name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = GatedCNN(x, 64, 5, stage=1, block='a')
    x = GatedCNN(x, 64, 5, stage=1, block='b')
    x = MaxPooling1D(name='pool_1')(x)

    x = GatedCNN(x, 128, 3, stage=2, block='a')
    x = GatedCNN(x, 128, 3, stage=2, block='b')
    x = GatedCNN(x, 128, 3, stage=2, block='c')
    x = MaxPooling1D(name='pool_2')(x)

    x = GatedCNN(x, 256, 3, stage=3, block='a')
    x = GatedCNN(x, 256, 3, stage=3, block='b')
    x = GatedCNN(x, 256, 3, stage=3, block='c')
    x = MaxPooling1D(name='pool_3')(x)

#    x = Flatten()(x)
    x = GlobalMaxPooling1D()(x)
    x = concatenate([x, x_c_c, embedding_2_d_1, embedding_2_d_2])
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = concatenate([x, x_wide])
    output = Dense(1, activation='sigmoid')(x)

    model= Model(inputs=[x_t_c, x_c_c, x_c_d_1, x_c_d_2, x_wide], outputs=output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()

