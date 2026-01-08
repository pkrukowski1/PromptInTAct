#!/bin/bash
#SBATCH --job-name=L2P_dil_imagenet-r_15_tasks_intactpp
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx


source activate prompt_intact_plus_plus

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/15-task/intactpp

# hard coded inputs
GPUID='0'
CONFIG=configs/intactpp/dil_imnet-r_prompt_15_tasks.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
LAMBDA_VAR_SCALES=("0.001" "0.01" "0.1" "1.0")
LAMBDA_DRIFT_SCALES=("0.001" "0.01" "0.1" "1.0")
LAMBDA_SLOPE_SCALES=("0.0001" "0.001" "0.1")


for var in "${LAMBDA_VAR_SCALES[@]}"; do
  for drift in "${LAMBDA_DRIFT_SCALES[@]}"; do
    for slope in "${LAMBDA_SLOPE_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/coda-p/var${var}_slope${slope}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name L2P \
            --prompt_param 30 20 -1 \
            --reg_type intactpp \
            --log_dir $LOGDIR \
            --lambda_var $var \
            --lambda_drift $drift \
            --lambda_slope_reg $slope \
            --n_last_blocks_to_finetune 0
    done
  done
done