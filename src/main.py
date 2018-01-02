#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
from glob import glob
from importlib import import_module
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import plot_model

from utils import parse_args, get_report, draw_curve
from tf_backend import *

def main():
    FLAGS = parse_args()
    print(FLAGS)

    if FLAGS.mode == 'train':
        print('-------------')
        print(' mode: train')
        print('-------------')

        print('load data...')
        input_data = h5py.File(FLAGS.data)

        print('load batch generator...')
        batch_module = import_module(FLAGS.batch_generator)
        generator = batch_module.BatchGenerator(FLAGS.data)

        print('load network struct...')
        model_module = import_module(FLAGS.network_struct)
        model = model_module.build_model(FLAGS.time_dim)
        
        print('training...')
        mc = ModelCheckpoint('%s.{epoch:03d}-{val_loss:.5f}.hdf5'%FLAGS.model,
                             'val_loss', save_best_only=True)
        es = EarlyStopping('val_loss', patience=200)
        history = model.fit_generator(generator.train(FLAGS.batch_size),
                                      steps_per_epoch=1000,
                                      epochs=1000,
                                      callbacks=[mc, es],
                                      validation_data=generator.valid())

        print('load best model...')
        model_path = sorted(glob(FLAGS.model+'*'))[-1]
        model = load_model(model_path)

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

        get_report(pred, true, FLAGS.report)
        draw_curve(history, FLAGS.curve)
        plot_model(model, FLAGS.struct_pic, show_shapes=True, show_layer_names=True)

        with open(FLAGS.predict, 'w') as file:
            for line in pred:
                file.write('{}\n'.format(line))

    else:
        print('---------------')
        print(' mode: predict')
        print('---------------')

        print('load batch generator...')
        batch_module = import_module(FLAGS.batch_generator)
        generator = batch_module.BatchGenerator(FLAGS.data)

        print('load model...')
        model = load_model(FLAGS.model)

        print('predict...')
        pred = []
        true = []
        test = generator.test(10000)
        for test_x, test_y in test:
            pred.append(model.predict_on_batch(test_x).flatten())
            true.append(test_y)
        pred = np.concatenate(pred)
        true = np.concatenate(true)

        if FLAGS.report != 'None':
            get_report(pred, true, FLAGS.report)
        if FLAGS.predict != 'None':
            with open(FLAGS.predict, 'w') as file:
                for line in pred:
                    file.write('{}\n'.format(line))
    

if __name__ == '__main__':
    main()
