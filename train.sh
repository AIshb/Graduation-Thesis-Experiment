#!/bin/bash

config_file=$1
export CUDA_VISIBLE_DEVICES="$2"

name=`echo $config_file | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

source config/env.sh
source $config_file

if [ -z $time_dim ]; then
    time_dim=12
fi

rm -f ${model}.*
rm -f ${report}
rm -f ${predict}
rm -f ${curve}_loss.png ${curve}_acc.png
rm -f ${struct_pic}

cd src
begin_time=`date "+%Y%m%d %H:%M:%S"`
python3 main.py \
    --mode train \
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
if [ $? -ne 0 ]; then
    echo "error occur!!"
    exit 1
fi
end_time=`date "+%Y%m%d %H:%M:%S"`

echo "begin_time: $begin_time"
echo "end_time: $end_time"
