#!/bin/bash

batch_size=128
time_dim=12
data=$data_dir/shuffle_n0.hdf5
network_struct=model.cnn_9_n0
batch_generator=batch.cnn_month_n0
model=$model_dir/$name
report=$report_dir/$name
predict=$predict_dir/$name
curve=$curve_dir/$name
struct_pic=$struct_pic_dir/${name}.png

