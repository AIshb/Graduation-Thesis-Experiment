#!/bin/bash

set -e

gpuid=$1

source config/env.sh

# for model_prefix in cnn_9_month cnn_10_month cnn_9_day cnn_10_day
for model_prefix in cnn_9_all cnn_10_all
do
    models=`ls ${model_dir}/${model_prefix}.* | tail -5 | awk -F '/' '{print $NF}' | awk -F '.' '{print $1"."$2"."$3}'`
    for model in $models
    do
        echo $model
        ./predict.sh config/predict/${model_prefix}.sh ${model_dir}/${model}.hdf5 ${CHURN_HOME}/ensemble/predict_proba/${model} ${gpuid}
        if [ $? -ne 0 ]; then
            echo "error occur!"
            exit 1
        fi
    done
done
