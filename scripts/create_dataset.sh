#!/bin/bash -e

INPUT_DIR=$1
DATASET_NAME=${2:-dataset}
COUNT=${3:-3000}

if [ ${INPUT_DIR:-UNDEF} == 'UNDEF' ]; then
    echo "usage: $0 <input directory> (<dataset name>)"
    exit 1
fi
if [ ! -d ${INPUT_DIR} ]; then
    echo "$INPUT_DIR should be a directory"
    exit 1
fi
if [ ! -d ${INPUT_DIR}/raw -o ! -d ${INPUT_DIR}/wide -o ! -d ${INPUT_DIR}/crop ]; then
    echo "$INPUT_DIR is expected to be created by preprocess_captures.py --extract_node_image"
    exit 1
fi

SCRIPT_DIR=$(cd `dirname $0` && pwd)
WORKING_DIR=$SCRIPT_DIR/work
DATASET_DIR=$WORKING_DIR/$DATASET_NAME

preprocess=$SCRIPT_DIR/preprocess_captures.py
copy_images=$SCRIPT_DIR/copy_images.sh

function create_classnumlist()
{
    input=$1
    python $preprocess --extract_label $input | sort | uniq -c | sort -nr | grep -v Namespace
}

function remove_invalid_images()
{
    input=$1
    find $input -name '*.png' -size -100c | xargs rm -f
    find $input -name '*.jpg' -size -100c | xargs rm -f
    find $input -name '*.jpg' | xargs identify -format "%i %w %h\n" | awk '$2 < 16 || $3 < 16 {print $0}' | xargs rm -f
}

if [ -e $WORKING_DIR ]; then
    echo "$WORKING_DIR has already existed. please remove work dir before"
    # exit
fi
mkdir -p $WORKING_DIR
mkdir -p $DATASET_DIR

if [ ! -d $WORKING_DIR/venv ]; then
    python3 -m venv $WORKING_DIR/venv
fi
source $WORKING_DIR/venv/bin/activate
pip install opencv-python tqdm

echo 'remove invalid png and jpg'
remove_invalid_images $INPUT_DIR

echo 'create class list'
create_classnumlist $INPUT_DIR > $WORKING_DIR/classnumlist
cat $WORKING_DIR/classnumlist | awk '{print $2}' > $WORKING_DIR/classlist
cat $WORKING_DIR/classnumlist | awk '$1 < 60 {print $2}' > $WORKING_DIR/classlist_low

echo 'extract images for augmentation'
$copy_images $WORKING_DIR/classlist $INPUT_DIR ${DATASET_DIR}_aug $((COUNT/12))

echo 'execute augmentation'
python $preprocess --augmentation resize --target_label $WORKING_DIR/classlist ${DATASET_DIR}_aug
python $preprocess --augmentation gamma --target_label $WORKING_DIR/classlist_low ${DATASET_DIR}_aug
python $preprocess --augmentation gamma --target_label $WORKING_DIR/classlist_low $WORKING_DIR/aug_resize

echo 'create dataset'
$copy_images $WORKING_DIR/classlist $INPUT_DIR/raw $DATASET_DIR $((COUNT/6))
$copy_images $WORKING_DIR/classlist $INPUT_DIR/wide $DATASET_DIR $((COUNT/6))
$copy_images $WORKING_DIR/classlist $INPUT_DIR/crop $DATASET_DIR $((COUNT/6))
$copy_images $WORKING_DIR/classlist $INPUT_DIR/additional $DATASET_DIR $((COUNT/6))
$copy_images $WORKING_DIR/classlist $WORKING_DIR/aug_resize $DATASET_DIR $((COUNT/3))
$copy_images $WORKING_DIR/classlist $WORKING_DIR/aug_gamma $DATASET_DIR $((COUNT/3))

echo 'remove broken images'
python $preprocess --check_jpg $DATASET_DIR | grep -v Namespace | xargs rm -f
remove_invalid_images $DATASET_DIR

echo 'separate validation data for efficientnet training'
rm -rf $WORKING_DIR/dataset_imagenet
python $preprocess --convert_imagenet --dest_dir $DATASET_DIR --validation_ratio 0.1 $DATASET_DIR
python $preprocess --extract_label $DATASET_DIR/train | sort | uniq > $DATASET_DIR/classlist
python $preprocess --extract_label $DATASET_DIR/validation | sort > $DATASET_DIR/synset_labels.txt

# echo 'create csv file for automl'
# python $preprocess --csv $DATASET_DIR

