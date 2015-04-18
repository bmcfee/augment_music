#!/bin/bash
#
# Train, validate, and eval a multi-instrument tagger.
#
# Requires the following:
#    - An environment variable `WORK` has been set, pointing to the expected
#      directory structure of data.
#
#
# Sample call:
# - Train an XXL model with the 0th augmentation index on the 4th fold.
#
#   ./sorcery.sh 0 xxlarge 3 train
#

BASEDIR=${WORK}
SRC=${HOME}/src/augment_music/code
cd SRC

# Directory of features, blah blah
FEATURES=${BASEDIR}/augmentation_features
METADATA=${SRC}/../data
MODELS=${BASEDIR}/augmentation_models
RESULTS=${BASEDIR}/results

MODEL_FILE="model_file.json"
SPLIT_FILE="train_test.json"

NUM_CPUS=1

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "sorcery {aug_idx} {size} {[0-4]|*all} {fit|select|evaluate|*all}"
    echo $'\taug_idx - Augmentation condition, in [0, 5].'
    echo $'\tsize - Model size, one of {small, medium, large, xlarge, xxlarge}'
    echo $'\tfold# - Index of the training fold to run, default=all.'
    echo $'\tphase - Name of training phase to run, default=all.'
    exit 0
fi

AUG_IDX="$1"

if [ -z "$2" ];
then
    MODEL_SIZE="xxlarge"
else
    MODEL_SIZE=$2
fi

if [ -z "$3" ] || [ "$3" == "all" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$3
fi

if [ -z "$4" ];
then
    PHASE="all"
else
    PHASE=$4
fi

EXP_NAME="fixme"

# Fit networks
# TODO(bmcfee): Any bright ideas on indexing the -t argument via AUG_IDX?
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
        ./train_optimus_model.py \
-i ${FEATURES} \
-t ${METADATA}/medley_index_1_nopitch.csv \
-a ${METADATA}/medley_artist_index.json \
-f ${idx} \
-s ${MODEL_SIZE} \
-n ${EXP_NAME} \
-o ${MODELS}/aug${AUG_IDX}/
    done
}
}
fi

# Validation Sweep
if [ $PHASE == "all" ] || [ $PHASE == "select" ];
then
    for idx in ${FOLD_IDXS}
    do
        for param_file in {??}
        do
            ./evaluate_model.py \
-i ${FEATURES} \
-t ${METADATA}/medley_index_1_nopitch.csv \
-s ${MODELS}/aug${AUG_IDX}/fold_0${idx}/${SPLIT_FILE} \
-n validation \
-j 1 \
-p ${param_file} \
-m ${MODELS}/aug${AUG_IDX}/fold_0${idx}/${MODEL_FILE} \
-o ${RESULTS}/aug${AUG_IDX}/fold_0${idx}/${param_file}.json
        done
    done
fi

# Identify best and symlink
# ??
FINAL_PARAMS=${param_file}

# Evaluate
if [ $PHASE == "all" ] || [ $PHASE == "evaluate" ];
then
    for idx in ${FOLD_IDXS}
    do
        for name in train test
        do
            ./evaluate_model.py \
-i ${FEATURES} \
-t ${METADATA}/medley_index_1_nopitch.csv \
-s ${MODELS}/aug${AUG_IDX}/fold_0${idx}/${SPLIT_FILE} \
-n ${NAME} \
-j ${NUM_CPUS} \
-p ${FINAL_PARAMS} \
-m ${MODELS}/aug${AUG_IDX}/fold_0${idx}/${MODEL_FILE} \
-o ${RESULTS}/aug${AUG_IDX}/fold_0${idx}/${name}.json
    done
fi
