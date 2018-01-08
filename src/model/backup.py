#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import add, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.h5_cnn_month import BatchGenerator, get_report
from .tools import get_report, draw_loss

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

def build_model():
    batch_size = 128
    month_size = 12
    cur_filename = __file__.split('/')[-1].split('.')[0] if len(sys.argv) < 4 else sys.argv[3]
    generator = BatchGenerator(sys.argv[1])

    input_1_c = Input(shape=(month_size, 50), dtype='float32', name='input_1_c')
    input_1_d = Input(shape=(month_size,), dtype='int32', name='input_1_d')
    input_2_c = Input(shape=(2,), dtype='float32', name='input_2_c')
    input_2_d_1 = Input(shape=(1,), dtype='int32', name='input_2_d_1')
    input_2_d_2 = Input(shape=(1,), dtype='int32', name='input_2_d_2')
    input_wide = Input(shape=(122,), dtype='float32', name='input_wide')

    embedding_1_d  = Embedding(input_dim=7+2, output_dim=5, input_length=month_size, name='embedding_1_d')(input_1_d)
    embedding_2_d_1 = Embedding(input_dim=4+2, output_dim=5, input_length=1, name='embedding_2_d_1')(input_2_d_1)
    embedding_2_d_2 = Embedding(input_dim=2+2, output_dim=5, input_length=1, name='embedding_2_d_2')(input_2_d_2)
    embedding_2_d_1 = Flatten(name='flatten_2_d_1')(embedding_2_d_1)
    embedding_2_d_2 = Flatten(name='flatten_2_d_2')(embedding_2_d_2)

    x = concatenate([input_1_c, embedding_1_d])
    x = BatchNormalization(name='bn_input')(x)
    x = Conv1D(64, 1, name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')

    x = Flatten()(x)
    x = concatenate([x, input_2_c, embedding_2_d_1, embedding_2_d_2])
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = concatenate([x, input_wide])
    output = Dense(1, activation='sigmoid')(x)

    model= Model(inputs=[input_1_c, input_1_d, input_2_c, input_2_d_1, input_2_d_2, input_wide], outputs=output)
    model.summary()
    plot_model(model, 'struct/%s.png'%cur_filename, show_shapes=True, show_layer_names=True)

    print('compile...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    mc = ModelCheckpoint('models/%s.{epoch:03d}-{val_loss:.5f}.hdf5'%cur_filename,
                         'val_loss', save_best_only=True)
    es = EarlyStopping('val_loss', patience=300)
    history = model.fit_generator(generator.train(batch_size),
                        steps_per_epoch=1000,
                        epochs=1000,
                        callbacks=[mc, es],
                        validation_data=generator.valid())

    print('predict...')
    pred = []
    true = []
    test = generator.test(10000)
    for test_x, test_y in test:
        pred.append(model.predict_on_batch(test_x).flatten())
        true.append(test_y)
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    print(pred.shape)
    print(true.shape)
    get_report(pred, true, 'report_%s.txt'%cur_filename)
    model.save('models/%s.hdf5'%cur_filename)

    with open('predict/%s.txt'%cur_filename, 'w') as file:
        for line in pred:
            file.write('{}\n'.format(line))
    
    print('save fig...')
    draw_loss(history)


if __name__ == '__main__':
    main()

