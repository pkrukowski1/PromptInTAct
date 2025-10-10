#!/bin/bash
# Offline (upper bound) training for ImageNet-R

DATASET=ImageNet_R
N_CLASS=200

OUTDIR=outputs/${DATASET}/offline
GPUID='0'
CONFIG=configs/imnet-r_ft.yaml
REPEAT=1
OVERWRITE=0

mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type default --learner_name NormalNN --upper_bound_flag \
    --log_dir ${OUTDIR}/normalnn