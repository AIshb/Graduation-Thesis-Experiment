#!/bin/bash

config_file=$1
name=`echo $config_file | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

source config/env.sh
source $config_file

tmp_dir=${CHURN_HOME}/tmp/${name}
log=./tmp/${name}.log

function update_best_model()
{
    rm -rf ${tmp_dir}/saved_model/*
    rm -rf ${tmp_dir}/report/*
    rm -rf ${tmp_dir}/predict_proba/*
    rm -rf ${tmp_dir}/curve/*
    rm -rf ${tmp_dir}/struct_pic/*
    cp ${model}.* ${tmp_dir}/saved_model/
    cp ${report} ${tmp_dir}/report/
    cp ${predict} ${tmp_dir}/predict_proba/
    cp ${curve}_loss.png ${tmp_dir}/curve/
    cp ${curve}_acc.png ${tmp_dir}/curve/
    cp ${struct_pic} ${tmp_dir}/struct_pic/
}

function save_best_model()
{
    rm -f ${model}.*
    rm -f ${report}
    rm -f ${predict}
    rm -f ${curve}_loss.png ${curve}_acc.png
    rm -f ${struct_pic}
    mv ${tmp_dir}/saved_model/* ${model_dir}/
    mv ${tmp_dir}/report/* ${report_dir}/
    mv ${tmp_dir}/predict_proba/* ${predict_dir}/
    mv ${tmp_dir}/curve/* ${curve_dir}/
    mv ${tmp_dir}/struct_pic/* ${struct_pic_dir}

    rm -rf $tmp_dir
}

rm -rf $tmp_dir
mkdir $tmp_dir
mkdir $tmp_dir/saved_model
mkdir $tmp_dir/report
mkdir $tmp_dir/predict_proba
mkdir $tmp_dir/curve
mkdir $tmp_dir/struct_pic

rm -f $log
echo $name > $log
echo "" >> $log

best_score=0
while true; do
    date "+%Y%m%d %H:%M:%S" >> $log

    ./train.sh $1 $2
    if [ $? -ne 0 ]; then
        echo "training terminated"
        echo "save best model..."
        echo "best precision: $best_score"
        if [ `echo "$best_score > 0" | bc` -eq 1 ]; then
            save_best_model
        fi
        break
    fi

    score=`tail -5 report/$name | head -1 | awk -F '\t' '{print $4}'`
    echo $score >> $log
    cat report/$name >> $log
    echo "" >> $Log

    if [ `echo "$score > $best_score" | bc` -eq 1 ]; then
        best_score=$score
        update_best_model        
    fi
done

