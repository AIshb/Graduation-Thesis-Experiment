#!/bin/bash

data=$data_dir/shuffle.hdf5
batch_generator=batch.cnn_month
model=$model_dir/cnn_9_s_imbalanced.038-0.11250.hdf5
#model=$model_dir/cnn_9_s.047-0.11260.hdf5
report=$CHURN_HOME/tmp/test
predict=None

