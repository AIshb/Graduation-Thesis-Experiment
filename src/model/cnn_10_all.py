#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import add, multiply, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam

def GatedCNN(input_tensor, nb_filter, kernel_size, time, stage, block):
    conv_name_base = time + '/' + 'conv' + str(stage) + block + '-'
    A = Conv1D(nb_filter, kernel_size, padding='same', name=conv_name_base+'a')(input_tensor)
    B = Conv1D(nb_filter, kernel_size, padding='same', name=conv_name_base+'b')(input_tensor)
    B_sig = Activation('sigmoid')(B)
    x = multiply([A, B_sig])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(*args, **kwargs):
    x_t_c_month = Input(shape=(12, 50), dtype='float32', name='month/x_t_c')
    x_t_c_day = Input(shape=(92, 34), dtype='float32', name='day/x_t_c')
    x_t_d_month = Input(shape=(12,), dtype='int32', name='month/x_t_d')
    x_c_c = Input(shape=(2,), dtype='float32', name='x_c_c')
    x_c_d_1 = Input(shape=(1,), dtype='int32', name='x_c_d_1')
    x_c_d_2 = Input(shape=(1,), dtype='int32', name='x_c_d_2')
    x_wide = Input(shape=(122,), dtype='float32', name='x_wide')

    embedding_1_d  = Embedding(input_dim=7+2, output_dim=5, input_length=12, name='embedding_1_d')(x_t_d_month)
    embedding_2_d_1 = Embedding(input_dim=4+2, output_dim=5, input_length=1, name='embedding_2_d_1')(x_c_d_1)
    embedding_2_d_2 = Embedding(input_dim=2+2, output_dim=5, input_length=1, name='embedding_2_d_2')(x_c_d_2)
    embedding_2_d_1 = Flatten(name='flatten_2_d_1')(embedding_2_d_1)
    embedding_2_d_2 = Flatten(name='flatten_2_d_2')(embedding_2_d_2)

    # month
    x = concatenate([x_t_c_month, embedding_1_d])
    x = BatchNormalization(name='month/bn_input')(x)
    x = Conv1D(64, 1, name='month/conv1')(x)
    x = BatchNormalization(name='month/bn_conv1')(x)
    x = Activation('relu')(x)

    x = GatedCNN(x, 64, 5, time='month', stage=1, block='a')
    x = GatedCNN(x, 64, 5, time='month', stage=1, block='b')
    x = MaxPooling1D(name='month/pool_1')(x)

    x = GatedCNN(x, 128, 3, time='month', stage=2, block='a')
    x = GatedCNN(x, 128, 3, time='month', stage=2, block='b')
    x = GatedCNN(x, 128, 3, time='month', stage=2, block='c')
    x = MaxPooling1D(name='month/pool_2')(x)

    x = GatedCNN(x, 256, 3, time='month', stage=3, block='a')
    x = GatedCNN(x, 256, 3, time='month', stage=3, block='b')
    x = GatedCNN(x, 256, 3, time='month', stage=3, block='c')
    x = MaxPooling1D(name='month/pool_3')(x)

    month_x = Flatten()(x)

    # day
    x = x_t_c_day
    x = BatchNormalization(name='day/bn_input')(x)
    x = Conv1D(64, 1, name='day/conv1')(x)
    x = BatchNormalization(name='day/bn_conv1')(x)
    x = Activation('relu')(x)

    x = GatedCNN(x, 64, 5, time='day', stage=1, block='a')
    x = GatedCNN(x, 64, 5, time='day', stage=1, block='b')
    x = MaxPooling1D(name='day/pool_1')(x)

    x = GatedCNN(x, 128, 3, time='day', stage=2, block='a')
    x = GatedCNN(x, 128, 3, time='day', stage=2, block='b')
    x = GatedCNN(x, 128, 3, time='day', stage=2, block='c')
    x = MaxPooling1D(name='day/pool_2')(x)

    x = GatedCNN(x, 256, 3, time='day', stage=3, block='a')
    x = GatedCNN(x, 256, 3, time='day', stage=3, block='b')
    x = GatedCNN(x, 256, 3, time='day', stage=3, block='c')
    x = MaxPooling1D(name='day/pool_3')(x)

    day_x = Flatten()(x)

    # concat
    x = concatenate([month_x, day_x, x_c_c, embedding_2_d_1, embedding_2_d_2])
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = concatenate([x, x_wide])
    output = Dense(1, activation='sigmoid')(x)

    model= Model(inputs=[x_t_c_month, x_t_c_day, x_t_d_month, x_c_c, x_c_d_1, x_c_d_2, x_wide], outputs=output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()

