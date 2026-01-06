#!/bin/bash
#SBATCH --job-name=DualPrompt_CIFAR100
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate prompt_intact

# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/intact/cifar-100_prompt.yaml
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
VAR_LOSS_SCALES=("0.001" "0.01" "0.1" "1.0")
INTERNAL_REPRESENTATION_DRIFT_REG_SCALES=("0.0")
FEATURE_LOSS_SCALES=("0.0001" "0.001" "0.1")

for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    for drift in "${FEATURE_LOSS_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/dual_prompt/var${var}_out${out}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name DualPrompt \
            --prompt_param 10 20 6 \
            --use_intact_regularization \
            --log_dir $LOGDIR \
            --var_loss_scale $var \
            --internal_repr_drift_loss_scale $out \
            --feature_loss_scale $drift \
            --use_align_loss
    done
  done
done