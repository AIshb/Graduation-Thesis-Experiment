#!/bin/bash

name=cnn_9_s_imbalanced

batch_size=128
time_dim=12
data=$data_dir/shuffle.hdf5
network_struct=model.cnn_9
batch_generator=batch.cnn_month_imbalanced
model=$model_dir/$name
report=$report_dir/$name
predict=$predict_dir/$name
curve=$curve_dir/$name
struct_pic=$struct_pic_dir/${name}.png

