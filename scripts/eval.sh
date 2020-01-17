#!/bin/bash -xe

MODEL_DIR=$1
TRAINING_LABELS=$2
TEST_DATASET_DIR=$3

script_dir=$(cd `dirname $0`; pwd)
test_dataset_name=$(basename $TEST_DATASET_DIR)
subdirs=$(find $TEST_DATASET_DIR -mindepth 1 -maxdepth 1 -type d)

for dir in ${subdirs[@]}
do
    type=$(basename $dir)
    python $script_dir/predict.py --ckpts_dir $MODEL_DIR/archive --training_labels $TRAINING_LABELS --image_dir $dir > $MODEL_DIR/result_${test_dataset_name}_${type}
done
