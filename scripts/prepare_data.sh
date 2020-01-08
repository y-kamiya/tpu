#!/bin/bash -xe

ROOT_DIR=$1
DATASET_NAME=$2
PROJECT_ID=$3

DATA_DIR=$ROOT_DIR/$DATASET_NAME
OUTPUT_DIR=$DATA_DIR/tf_records

if [ ! -e $DATA_DIR ]; then
    echo "not found $DATA_DIR"
    exit
fi

docker run -v $ROOT_DIR:/root -w /root -it tensorflow/tensorflow:1.14.0 bash -c "pip install gcloud google-cloud-storage && python /root/tools/datasets/imagenet_to_gcs.py  --raw_data_dir /root/$DATASET_NAME --local_scratch_dir /root/$DATASET_NAME/tf_records --nogcs_upload"

gsutil -m cp -r $OUTPUT_DIR/train/train-*  gs://$PROJECT_ID-vcm/$DATASET_NAME/tf_records/
gsutil -m cp -r $OUTPUT_DIR/validation/validation-*  gs://$PROJECT_ID-vcm/$DATASET_NAME/tf_records/
