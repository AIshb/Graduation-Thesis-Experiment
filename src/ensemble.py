#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn import metrics

def get_report(proba, label, report_file_path):
    sorted_index = np.argsort(-proba)

    proba = proba[sorted_index]
    label = label[sorted_index]

    valid_size = [10000, 15000, 20000, 25000, 50000, 100000, 150000, 200000]

    p,r,t = precision_recall_curve(label, proba)
    pr_auc = metrics.auc(r,p)
    auc = roc_auc_score(label, proba)

    report = open(report_file_path, 'w')

    print('Total\t%d\nChurn\t%d\nchurn_rate\t%5.4f\nAUC\t%5.4f\npr_AUC\t%5.4f' % (label.size, sum(label), float(sum(label))/label.size, auc, pr_auc))
#    report.write('Total\t%d\nChurn\t%d\nchurn_rate\t%5.4f\nAUC\t%5.4f\npr_AUC\t%5.4f\n' % (label.size, sum(label), float(sum(label))/label.size, auc, pr_auc))

#    report.write('Top\tChurn\tRecall\tPrecision\n')
    print('Top\tChurn\tRecall\tPrecision')

    precision_25000 = 0.0
    for size in valid_size:
        predict = np.zeros(label.size)
        predict[:size] = 1
        recall = recall_score(label, predict, average='binary')
        precision = precision_score(label, predict, average='binary')
        if size == 25000:
            precision_25000 = precision
#        report.write('%d\t%d\t%5.4f\t%5.4f\n' % (size, sum(label[:size]), recall, precision))
        print('%d\t%d\t%5.4f\t%5.4f' % (size, sum(label[:size]), recall, precision))

    return precision_25000

def run(preds, true):
    max_precision = 0.0
    best_idx = None
    idxs = np.arange(preds.shape[1])
    for i in range(1, preds.shape[1]+1):
        sub_idxs = combinations(idxs, i)
        for idx in sub_idxs:
            print(idx)
            pred = np.mean(preds[:, idx], axis=1)
            precision = get_report(pred, true, 'tmp/report.txt')
            print(precision)
            if max_precision < precision:
                max_precision = precision
                best_idx = idx
            print('')
    print(best_idx)
    print(max_precision)

def main():
    preds = []
    true = pd.read_csv(sys.argv[1], header=None)
    for name in sys.argv[2:]:
        print('load {}...'.format(name))
        preds.append(pd.read_csv(name, header=None))
    preds = pd.concat(preds, axis=1)
    run(preds.values, true.values)


if __name__ == '__main__':
    main()

