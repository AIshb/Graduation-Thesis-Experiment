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

from utils import get_report, draw_curve
from tf_backend import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('-td', '--time_dim', type=int,
                        default=12, help='time_dim')
    parser.add_argument('-d', '--data', type=str,
                        help='input data')
    parser.add_argument('-ns', '--network_struct', type=str,
                        help='network model file')
    parser.add_argument('-bg', '--batch_generator', type=str,
                        help='batch generator file')
    parser.add_argument('-m', '--model', type=str,
                        help='final model')
    parser.add_argument('-r', '--report', type=str,
                        help='report about model performance')
    parser.add_argument('-p', '--predict', type=str,
                        help='predict result')
    parser.add_argument('-c', '--curve', type=str,
                        help='loss and acc curve')
    parser.add_argument('-sp', '--struct_pic', type=str,
                        help='model struct')

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

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

#    model.save(FLAGS.model)
    get_report(pred, true, FLAGS.report)
    draw_curve(history, FLAGS.curve)
    plot_model(model, FLAGS.struct_pic, show_shapes=True, show_layer_names=True)

    with open(FLAGS.predict, 'w') as file:
        for line in pred:
            file.write('{}\n'.format(line))

    
if __name__ == '__main__':
    main()
