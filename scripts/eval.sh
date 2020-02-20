#!/bin/bash -xe

MODEL_DIR=$1
TRAINING_LABELS=$2
TEST_DATASET_DIR=$3

script_dir=$(cd `dirname $0`; pwd)
working_dir=$script_dir/work
test_dataset_name=$(basename $TEST_DATASET_DIR)
subdirs=$(find $TEST_DATASET_DIR -mindepth 1 -maxdepth 1 -type d)

if [ ! -d $working_dir/venv ]; then
    python3 -m venv $working_dir/venv
fi
source $working_dir/venv/bin/activate
pip install tensorflow==1.15.0

for dir in ${subdirs[@]}
do
    type=$(basename $dir)
    python $script_dir/preprocess_captures.py --extract_label $dir | sort > $dir/synset_labels.txt
    python $script_dir/predict.py --ckpts_dir $MODEL_DIR/archive --training_labels $TRAINING_LABELS --image_dir $dir > $MODEL_DIR/result_${test_dataset_name}_${type}
done

deactivate
