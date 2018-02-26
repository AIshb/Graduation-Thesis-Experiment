#!/bin/bash

config_file=$1

name=`echo $config_file | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
log=tmp/choose_high_score.log

rm -f $log
echo $name > $log
echo "" >> $log

while true; do
    date "+%Y%m%d %H:%M:%S" >> $log

    ./train.sh $1 $2
    if [ $? -ne 0 ]; then
        echo "error!!!"
        break
    fi

    score=`tail -5 report/$name | head -1 | awk -F '\t' '{print $4}'`
    echo $score >> $log
    echo "" >> $Log

    if [ `echo "$score > 0.53" | bc` -eq 1 ]; then
        break
    fi
done

