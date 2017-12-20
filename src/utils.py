#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_report(proba, label, report_file_path):
    sorted_index = np.argsort(-proba)

    proba = proba[sorted_index]
    label = label[sorted_index]

    valid_size = [10000, 15000, 20000, 25000, 50000, 100000, 150000, 200000]

    p,r,t = precision_recall_curve(label, proba)
    pr_auc = metrics.auc(r,p)
    auc = roc_auc_score(label, proba)

    report = open(report_file_path, 'w')

    report.write('Total\t%d\nChurn\t%d\nchurn_rate\t%5.4f\nAUC\t%5.4f\npr_AUC\t%5.4f\n' % (label.size, sum(label), float(sum(label))/label.size, auc, pr_auc))

    report.write('Top\tChurn\tRecall\tPrecision\n')
    print('Top\tChurn\tRecall\tPrecision')

    for size in valid_size:
        predict = np.zeros(label.size)
        predict[:size] = 1
        recall = recall_score(label, predict, average='binary')
        precision = precision_score(label, predict, average='binary')
        report.write('%d\t%d\t%5.4f\t%5.4f\n' % (size, sum(label[:size]), recall, precision))
        print('%d\t%d\t%5.4f\t%5.4f' % (size, sum(label[:size]), recall, precision))

def draw_curve(history, filename):
    x = history.epoch
    y_loss = history.history['loss']
    y_val_loss = history.history['val_loss']
    y_acc = history.history['acc']
    y_val_acc = history.history['val_acc']

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_loss, c='r', label='loss')
    plt.plot(x, y_val_loss, c='b', label='val_loss')
    plt.legend()
    plt.savefig('{}_loss.png'.format(filename), dpi=200)
    plt.close()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(x, y_acc, c='r', label='acc')
    plt.plot(x, y_val_acc, c='b', label='val_acc')
    plt.legend()
    plt.savefig('{}_acc.png'.format(filename), dpi=200)

