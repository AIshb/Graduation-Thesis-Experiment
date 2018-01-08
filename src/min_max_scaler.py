#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np

def main():
    in_f = h5py.File(sys.argv[1])
    out_f = h5py.File(sys.argv[2], 'w')

    print('month_x_t_c')
    month_x_t_c = np.concatenate([in_f['train/month/x_t_c'], in_f['valid/month/x_t_c'], in_f['test/month/x_t_c']])
    print('day_x_t_c')
    day_x_t_c = np.concatenate([in_f['train/day/x_t_c'], in_f['valid/day/x_t_c'], in_f['test/day/x_t_c']])
    print('x_c_c')
    x_c_c = np.concatenate([in_f['train/x_c_c'], in_f['valid/x_c_c'], in_f['test/x_c_c']])

    print('min_max')
    min_month_x_t_c = np.min(month_x_t_c, axis=(0, 1))
    max_month_x_t_c = np.max(month_x_t_c, axis=(0, 1))
    min_day_x_t_c = np.min(day_x_t_c, axis=(0, 1))
    max_day_x_t_c = np.max(day_x_t_c, axis=(0, 1))
    min_x_c_c = np.min(x_c_c, axis=0)
    max_x_c_c = np.max(x_c_c, axis=0)

    del month_x_t_c
    del day_x_t_c
    del x_c_c

    for name in ('train', 'valid', 'test'):
        print(name)
        in_group = in_f[name]
        out_group = out_f.create_group(name)

        for data in ('month/x_t_d', 'x_c_d_1', 'x_c_d_2', 'x_wide', 'y'):
            out_group.create_dataset(data, data=in_group[data])

        month_x_t_c = (in_group['month/x_t_c'] - min_month_x_t_c) / (max_month_x_t_c - min_month_x_t_c)
        out_group.create_dataset('month/x_t_c', data=month_x_t_c)
        del month_x_t_c

        day_x_t_c = (in_group['day/x_t_c'] - min_day_x_t_c) / (max_day_x_t_c - min_day_x_t_c)
        out_group.create_dataset('day/x_t_c', data=day_x_t_c)
        del day_x_t_c

        x_c_c = (in_group['x_c_c'] - min_x_c_c) / (max_x_c_c - min_x_c_c)
        out_group.create_dataset('x_c_c', data=x_c_c)
        del x_c_c

    out_f.close()


if __name__ == '__main__':
    main()

