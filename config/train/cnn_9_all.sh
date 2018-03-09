#!/bin/bash

batch_size=128
data=$data_dir/shuffle.hdf5
network_struct=model.cnn_9_all
batch_generator=batch.cnn_all
model=$model_dir/$name
report=$report_dir/$name
predict=$predict_dir/$name
curve=$curve_dir/$name
struct_pic=$struct_pic_dir/${name}.png

