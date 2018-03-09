#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

class BatchGenerator:
    def __init__(self, filename):
        self.file = h5py.File(filename, 'r')
        self.feature_names = ('month/x_t_c', 'month/x_t_d', 'day/x_t_c', 'x_c_c', 'x_c_d_1', 'x_c_d_2', 'x_wide')
        self.train_x = {key: self.file['train/'+key].value for key in self.feature_names}
        self.train_y = self.file['train/y'].value
        self.valid_x = {key: self.file['valid/'+key].value for key in self.feature_names}
        self.valid_y = self.file['valid/y'].value

    def train(self, batch_size=128):
        size = self.train_y.shape[0]
        idx = np.arange(size)
        
        while (True):
            np.random.shuffle(idx)
            for i in range(0, size, batch_size):
                begin = i
                end = begin + batch_size
                
                if end > size:
                    x = {key: self.train_x[key][idx[-batch_size:]] for key in self.feature_names}
                    y = self.train_y[idx[-batch_size:]]
                else:
                    x = {key: self.train_x[key][idx[begin:end]] for key in self.feature_names}
                    y = self.train_y[idx[begin:end]]

                if sum(y==1) == 0:
                    continue

                yield (x, y)

    def test(self, batch_size=200000):
        test_data = self.file['test']
        
        for i in range(0, test_data['y'].shape[0], batch_size):
            begin = i
            end = begin + batch_size
            
            x = {key: test_data[key][begin:end] for key in self.feature_names}
            y = test_data['y'][begin:end]
            yield (x, y)

    def valid(self):
        return (self.valid_x, self.valid_y)

