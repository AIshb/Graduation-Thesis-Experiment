#!/bin/bash

config_file=$1
export CUDA_VISIBLE_DEVICES="$2"

source config/env.sh
source $config_file

rm -f ${report}
rm -f ${predict}

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
