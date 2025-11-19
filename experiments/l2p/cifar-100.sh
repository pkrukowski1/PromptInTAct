#!/bin/bash
#SBATCH --job-name=L2P_CIFAR100
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

# TO RUN LOCALLY
# OUTDIR=./${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cifar-100_prompt.yaml
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
var_loss_scaleS=("0.001" "0.01" "0.1" "1.0")
internal_repr_drift_loss_scaleS=("0.001" "0.01" "1.0")
FEATURE_LOSS_SCALES=("0.0001" "0.001" "0.1")

for var in "${var_loss_scaleS[@]}"; do
  for out in "${internal_repr_drift_loss_scaleS[@]}"; do
    for drift in "${FEATURE_LOSS_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/l2p/var${var}_out${out}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name L2P \
            --prompt_param 30 20 -1 \
            --use_interval_activation \
            --log_dir $LOGDIR \
            --var_loss_scale $var \
            --internal_repr_drift_loss_scale $out \
            --feature_loss_scale $drift \
            --use_align_loss
    done
  done
done