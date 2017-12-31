#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

def main():
    num = 1415429
    idx = np.arange(num)
    np.random.shuffle(idx)
    train_idx = idx[:int(np.around(num*0.8))]
    val_idx = idx[-int(np.around(num*0.8)):]
    train_idx = set(train_idx)
    val_idx = set(val_idx)

    train = open(sys.argv[2], 'w')
    val = open(sys.argv[3], 'w')
    with open(sys.argv[1]) as file:
        for i, line in enumerate(file):
            line = line.strip()
            if i in train_idx:
                train.write(line+'\n')
            elif i in val_idx:
                val.write(line+'\n')
    train.close()
    val.close()


if __name__ == '__main__':
    main()

