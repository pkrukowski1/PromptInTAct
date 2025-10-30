#!/bin/bash
#SBATCH --job-name=L2P_imagenet-r_5_tasks_dil_baseline
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/5-task_dil_baseline

# hard coded inputs
GPUID='0'
CONFIG=configs/dil_imnet-r_prompt_5_tasks.yaml
REPEAT=5
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

LOGDIR=${OUTDIR}/l2p/var${var}_out${out}_drift${drift}
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir $LOGDIR