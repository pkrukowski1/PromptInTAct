#!/bin/bash
#SBATCH --job-name=CODA-P_domainnet_baseline_dil
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
OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/6-task_baseline

# hard coded inputs
GPUID='0'
CONFIG=configs/domainnet_prompt_dil.yaml
REPEAT=3
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

LOGDIR=${OUTDIR}/coda-p/dil_baseline
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --log_dir $LOGDIR