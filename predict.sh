#!/bin/bash

export CUDA_VISIBLE_DEVICES="$4"

source config/env.sh

rm -f ${report}
rm -f ${predict}

config_file=$1
model=$2
predict=$3

source $config_file

rm -f $predict
echo "-------------------"
echo "predict ${model}..."
echo "-------------------"

cd src
begin_time=`date "+%Y%m%d %H:%M:%S"`
python3 main.py \
    --mode predict \
    --data $data \
    --batch_generator $batch_generator \
    --model $model \
    --report $report \
    --predict $predict
end_time=`date "+%Y%m%d %H:%M:%S"`

echo "begin_time: $begin_time"
echo "end_time: $end_time"
