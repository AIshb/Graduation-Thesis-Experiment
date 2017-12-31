#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

from .tf_backend import *

class BatchGenerator:
    def __init__(self, filename):
        self.file = h5py.File(filename, 'r')
        self.feature_names = ('input_1_c', 'input_1_d', 'input_2_c', 'input_2_d_1', 'input_2_d_2', 'input_wide')
        self.train_x = {key: self.file['train/'+key].value for key in self.feature_names}
        self.train_y = self.file['train/y'].value
        self.valid_x = {key: self.file['valid/'+key].value for key in self.feature_names}
        self.valid_y = self.file['valid/y'].value

    def train(self, batch_size=128):
        idxs = [np.where(self.train_y==i)[0] for i in range(2)]
        size = [len(idxs[i]) for i in range(2)]
#        train_x = [self.train_x[idx[i]] for i in range(2)]
#        train_y = [self.train_y[idx[i]] for i in range(2)]
        batch = [batch_size-batch_size//5, batch_size//5]
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
            yield (x, y)

#    def train(self, batch_size=128):
#        size = self.train_y.shape[0]
#        idx = np.arange(size)
#        
#        while (True):
#            np.random.shuffle(idx)
#            for i in range(0, size, batch_size):
#                begin = i
#                end = begin + batch_size
#                
#                if end > size:
#                    x = {key: self.train_x[key][idx[-batch_size:]] for key in self.feature_names}
#                    y = self.train_y[idx[-batch_size:]]
#                else:
#                    x = {key: self.train_x[key][idx[begin:end]] for key in self.feature_names}
#                    y = self.train_y[idx[begin:end]]
#
#                if sum(y==1) == 0:
#                    continue
#
#                yield (x, y)

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

def get_report(proba, label, report_file_path):
    sorted_index = np.argsort(-proba)

    proba = proba[sorted_index]
    label = label[sorted_index]

    valid_size = [10000, 15000, 20000, 25000, 50000, 100000, 150000, 200000]

    p,r,t = precision_recall_curve(label, proba)
    pr_auc = metrics.auc(r,p)
    auc = roc_auc_score(label, proba)

    report = open(report_file_path, 'w')

    report.write('Total\t%d\nChurn\t%d\nchurn_rate\t%5.4f\nAUC\t%5.4f\npr_AUC\t%5.4f\n' % (label.size, sum(label),float(sum(label))/label.size, auc, pr_auc))

    report.write('Top\tChurn\tRecall\tPrecision\n')
    print('Top\tChurn\tRecall\tPrecision')

    for size in valid_size:
        predict = np.zeros(label.size)
        predict[0:size] = 1
        recall = recall_score(label, predict, average='binary')
        precision = precision_score(label, predict, average='binary')
        report.write('%d\t%d\t%5.4f\t%5.4f\n' % (size, sum(label[:size]), recall, precision))
        print('%d\t%d\t%5.4f\t%5.4f' % (size, sum(label[:size]), recall, precision))

