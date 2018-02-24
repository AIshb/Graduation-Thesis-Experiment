#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from keras.models import load_model

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tfb.set_session(session)

    model= load_model(sys.argv[1])
    model.summary()


if __name__ == '__main__':
    main()
