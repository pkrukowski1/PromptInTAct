#!/bin/bash
#SBATCH --job-name=L2P_imagenet-r_10_tasks
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx


source activate prompt_intact

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/intact/imnet-r_prompt_10_tasks.yaml
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
VAR_LOSS_SCALES=("0.001" "0.01" "0.1" "1.0")
INTERNAL_REPRESENTATION_DRIFT_REG_SCALES=("0.0")
FEATURE_LOSS_SCALES=("0.0001" "0.001" "0.1")

for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    for drift in "${FEATURE_LOSS_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/l2p/var${var}_out${out}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name L2P \
            --prompt_param 30 20 -1 \
            --use_intact_regularization \
            --log_dir $LOGDIR \
            --var_loss_scale $var \
            --internal_repr_drift_loss_scale $out \
            --feature_loss_scale $drift \
            --use_align_loss
    done
  done
done