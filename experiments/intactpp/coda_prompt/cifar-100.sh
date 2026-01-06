#!/bin/bash
#SBATCH --job-name=CODA-P_CIFAR100_intactpp
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate prompt_intact_plus_plus

# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/intactpp/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
# Define the ranges you want to search over
LAMBDA_VAR_SCALES=("0.001" "0.01" "0.1" "1.0")
LAMBDA_DRIFT_SCALES=("0.01" "0.1" "1.0")
LAMBDA_SLOPE_SCALES=("0.0001" "0.001" "0.1")

for var in "${LAMBDA_VAR_SCALES[@]}"; do
  for drift in "${LAMBDA_DRIFT_SCALES[@]}"; do
    for slope in "${LAMBDA_FEAT_SCALES[@]}"; do
      LOGDIR=${OUTDIR}/intactpp/coda-p/var${var}_slope${slope}_drift${drift}
      mkdir -p $LOGDIR
      python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name CODAPrompt \
        --prompt_param 100 8 0.0 \
        --use_intactpp_regularization \
        --log_dir $LOGDIR \
        --lambda_var $var \
        --lambda_drift $drift \
        --lambda_slope_reg $slope
    done
  done
done