#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    mode_parser = argparse.ArgumentParser()
    mode_parser.add_argument('-M', '--mode', type=str,
                            choices=['train', 'predict'], help='mode')
    FLAGS, unparsed = mode_parser.parse_known_args()
    
    parser = argparse.ArgumentParser()
    if FLAGS.mode == 'train':
        print('train')
        parser.add_argument('-M', '--mode', type=str,
                            choices=['train', 'predict'], help='mode')
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

    elif FLAGS.mode == 'predict':
        print('predict')
        parser.add_argument('-M', '--mode', type=str,
                            choices=['train', 'predict'], help='mode')
        parser.add_argument('-d', '--data', type=str,
                            help='input data')
        parser.add_argument('-bg', '--batch_generator', type=str,
                            help='batch generator file')
        parser.add_argument('-m', '--model', type=str,
                            help='model file')
        parser.add_argument('-r', '--report', type=str,
                            default='None', help='report about model performance')
        parser.add_argument('-p', '--predict', type=str,
                            default='None', help='predict result')

    FLAGS, unparsed = parser.parse_known_args()
    print('FLAGS:', FLAGS)
    print('unparsed:', unparsed)

    return FLAGS

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

