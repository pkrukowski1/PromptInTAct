#!/bin/bash
#SBATCH --job-name=DualPrompt_dil_imagenet-r_5_tasks_intactpp
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate intact_plus_plus

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/5-task/intactpp

# hard coded inputs
GPUID='0'
CONFIG=configs/intactpp/dil_imnet-r_prompt_5_tasks.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
LAMBDA_VAR_SCALES=("0.001" "0.01" "0.1" "1.0")
LAMBDA_DRIFT_SCALES=("0.001" "0.01" "0.1" "1.0")



for var in "${LAMBDA_VAR_SCALES[@]}"; do
  for drift in "${LAMBDA_DRIFT_SCALES[@]}"; do
    LOGDIR=${OUTDIR}/dual_prompt/var${var}_drift${drift}
    mkdir -p $LOGDIR
    python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name DualPrompt \
        --prompt_param 5 20 6 \
        --reg_type intactpp \
        --log_dir $LOGDIR \
        --lambda_var $var \
        --lambda_drift $drift \
        --n_last_blocks_to_finetune 0 
  done
done