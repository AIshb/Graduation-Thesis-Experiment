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
        self.feature_names = ('month/x_t_c', 'month/x_t_d', 'x_c_c', 'x_c_d_1', 'x_c_d_2', 'x_wide')
        self.train_x = {key: self.file['train/'+key].value for key in self.feature_names}
        self.train_y = self.file['train/y'].value
        self.valid_x = {key: self.file['valid/'+key].value for key in self.feature_names}
        self.valid_y = self.file['valid/y'].value

    def train(self, batch_size=128):
        idxs = [np.where(self.train_y==i)[0] for i in range(2)]
        size = [len(idxs[i]) for i in range(2)]
        batch = [batch_size-batch_size//10, batch_size//10] # 负例:正例=4:1
        begin = [0, 0]

        while (True):
            idx = []

            for i in range(2):
                end = begin[i] + batch[i]
                if end > size[i]:
                    idx.append(idxs[i][-batch[i]:])
                else:
                    idx.append(idxs[i][begin[i]:end])
                begin[i] = end
                if begin[i] >= size[i]:
                    begin[i] = 0
                    np.random.shuffle(idxs[i])
            idx = np.concatenate(idx)
            np.random.shuffle(idx) 

            x = {key: self.train_x[key][idx] for key in self.feature_names}
            y = self.train_y[idx]

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

