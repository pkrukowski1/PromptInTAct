#!/bin/bash
#SBATCH --job-name=L2P_domainnet_dil_baseline
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

# bash experiments/domainnet.sh
# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/5-task_dil

# hard coded inputs
GPUID='0'
CONFIG=configs/domainnet_prompt.yaml
REPEAT=3
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