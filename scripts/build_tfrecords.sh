#!/bin/bash

ROOT_DIR=$(cd $1 && pwd)
DATASET_DIR=$2
PROJECT_ID=$3


if [ ! -e $DATASET_DIR ]; then
    echo "not found $DATASET_DIR"
    exit
fi

output_dir=$DATASET_DIR/tf_records
dataset_name=$(basename $DATASET_DIR)

docker run -v $ROOT_DIR:/root -w /root -it tensorflow/tensorflow:1.15.0 bash -c "python /root/tools/datasets/imagenet_to_gcs.py  --raw_data_dir /root/$DATASET_DIR --local_scratch_dir /root/$output_dir --nogcs_upload"
if [ $? -ne 0 ]; then
    pip2.7 install tensorflow==1.15.0
    python2.7 $ROOT_DIR/tools/datasets/imagenet_to_gcs.py  --raw_data_dir $DATASET_DIR --local_scratch_dir $output_dir --nogcs_upload
fi

gsutil -m cp -r $output_dir/train/train-*  gs://${PROJECT_ID}-vcm/$dataset_name/tf_records/
gsutil -m cp -r $output_dir/validation/validation-*  gs://${PROJECT_ID}-vcm/$dataset_name/tf_records/
