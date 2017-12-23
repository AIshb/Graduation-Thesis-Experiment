#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

age_idx_dic = {}
age_idx_dic[3] = 0
age_idx_dic[1] = 1
age_idx_dic[2] = 12
age_list = [19, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]
for i in range(len(age_list)-1):
    for age in range(age_list[i], age_list[i+1]):
        age_idx_dic[age] = i + 2

# 122 features
def wide_feature(x):
    x = [x[3], age_idx_dic[x[2]], x[1], x[0]]
    encoder = OneHotEncoder(n_values=[7, 13, 4, 2], sparse=False)
    x_encoder = encoder.fit_transform([x])[0]
    feature_indices_ = encoder.feature_indices_
    
    result = []
    # terminal
    for i in range(1, 7):
        terminal = x_encoder[feature_indices_[0]+i]
        # age
        for j in range(1, 13):
            result.append(terminal*x_encoder[feature_indices_[1]+j])

        # gender
        for j in range(2):
            result.append(terminal*x_encoder[feature_indices_[2]+j])

        # is_shanghai
        for j in range(2):
            result.append(terminal*x_encoder[feature_indices_[3]+j])

    result += list(x_encoder)
    return result

def save_hdf5(group, filename, length):
    month_size = 12
    day_size = 92
    month_x_t_c = group.create_dataset('month/x_t_c', (length, month_size, 50))
    month_x_t_d = group.create_dataset('month/x_t_d', (length, month_size)) # 终端
    day_x_t_c = group.create_dataset('day/x_t_c', (length, day_size, 34))
    x_c_c = group.create_dataset('x_c_c', (length, 2)) # 入网时长、年龄
    x_c_d_1 = group.create_dataset('x_c_d_1', (length, 1), dtype='i') # 性别
    x_c_d_2 = group.create_dataset('x_c_d_2', (length, 1), dtype='i') # 是否上海人
    x_wide = group.create_dataset('x_wide', (length, 122))
    month_x_t_n0 = group.create_dataset('month/x_t_n0', (length, 1), dtype='i')
    day_x_t_n0 = group.create_dataset('day/x_t_n0', (length, 1), dtype='i')
    x_c_n0 = group.create_dataset('x_c_n0', (length, 1), dtype='i')
    y = group.create_dataset('y', (length,), dtype='i')

    with open(filename) as file:
        reader = csv.reader(file)

        for i, line in enumerate(reader):
            if i % 10000 == 9999:
                print(i+1)

            line = [v if v != '' else '-9999.0' for v in line]
            x = [float(v) for v in line[1:]]
            y[i] = int(float(line[0]))

            line_month_t_c = []
            line_month_t_d = []
            line_day_t_c = []
            line_c_c = [v if v != -9999.0 else 0 for v in [x[4], x[2]]]
            line_c_d_1 = [0 if x[1] == -9999.0 else int(x[1])+1]
            line_c_d_2 = [0 if x[0] == -9999.0 else int(x[0])+1]
            line_month_x_t_n0 = []
            line_day_x_t_n0 = []
            line_x_c_n0 = []

            # 月
            features = x[3:3+52*month_size]
            for j in range(month_size):
                start = j * 52
                end = (j + 1) * 52
                line_month_t_c.append([v if v != -9999.0 else 0 for v in features[start+2:end]])
                line_month_t_d.append(0 if x[start] == -9999.0 else int(x[start])+1)

            # 日
            features = x[-day_size*34:]
            for j in range(day_size):
                start = j * 34
                end = (j + 1) * 34
                line_day_t_c.append([v if v != -9999.0 else 0 for v in features[start:end]])

            # 统计0的数量
            line_month_x_t_n0.append(sum([1 if v == 0 else 0 for v in line_month_t_c]) \
                                    +sum([1 if v == 0 else 0 for v in line_month_t_d]))
            line_day_x_t_n0.append(sum([1 if v == 0 else 0 for v in line_day_t_c]))
            line_x_c_n0.append(sum(1 if v == 0 else 0 for v in line_c_c) \
                            +(1 if line_c_d_1[0] == 0 else 0) \
                            +(1 if line_c_d_2[0] == 0 else 0))

#            print(np.array(line_month_t_c).shape)
            month_x_t_c[i] = np.array(line_month_t_c)
            month_x_t_d[i] = np.array(line_month_t_d)
            day_x_t_c[i] = np.array(line_day_t_c)
            x_c_c[i] = np.array(line_c_c)
            x_c_d_1[i] = np.array(line_c_d_1)
            x_c_d_2[i] = np.array(line_c_d_2)
            x_wide[i] = np.array(wide_feature(x))
            month_x_t_n0[i] = np.array(line_month_x_t_n0)
            day_x_t_n0[i] = np.array(line_day_x_t_n0)
            x_c_n0[i] = np.array(line_x_c_n0)

def main():
    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file = sys.argv[3]
    train_len = 1132343
    valid_len = 283086
    test_len = 1380154
#    train_len = 10
#    valid_len = 10
#    test_len = 10
    
    f = h5py.File(sys.argv[4], 'w')

    print('train')
    group = f.create_group('train')
    save_hdf5(group, train_file, train_len)

    print('valid')
    group = f.create_group('valid')
    save_hdf5(group, valid_file, valid_len)

    print('test')
    group = f.create_group('test')
    save_hdf5(group, test_file, test_len)

    f.close()

if __name__ == '__main__':
    main()

