#!/bin/bash

batch_size=128
time_dim=92
data=$data_dir/shuffle.hdf5
network_struct=model.deep_9_day
batch_generator=batch.deep_day
model=$model_dir/$name
report=$report_dir/$name
predict=$predict_dir/$name
curve=$curve_dir/$name
struct_pic=$struct_pic_dir/${name}.png

