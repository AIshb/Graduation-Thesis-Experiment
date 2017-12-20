#!/bin/bash

config_file=$1

source config/env.sh
source $config_file

rm -f ${model}.*
rm -f ${report}
rm -f ${predict}
rm -f ${curve}_loss.png ${curve}_acc.png
rm -f ${struct_pic}

cd src
begin_time=`date "+%Y%m%d %H:%M:%S"`
python3 main.py \
    --batch_size $batch_size \
    --time_dim $time_dim \
    --data $data \
    --network_struct $network_struct \
    --batch_generator $batch_generator \
    --model $model \
    --report $report \
    --predict $predict \
    --curve $curve \
    --struct_pic $struct_pic
end_time=`date "+%Y%m%d %H:%M:%S"`

echo "begin_time: $begin_time"
echo "end_time: $end_time"
