#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import add, concatenate
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.optimizers import Adam

def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(nb_filter1, 1, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(nb_filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(nb_filter3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, stride=2):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(nb_filter1, 1, strides=stride, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(nb_filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(nb_filter3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1D(nb_filter3, 1, strides=stride, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
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

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')

    x = AveragePooling1D()(x)

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

