#!/bin/bash
#SBATCH --job-name=domainnet_dil_offline
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

DATASET=DomainNet
N_CLASS=345

OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/6-task_dil
GPUID='0'
CONFIG=configs/domainnet_prompt.yaml
REPEAT=3
OVERWRITE=0

mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type default --learner_name NormalNN --upper_bound_flag \
    --log_dir ${OUTDIR}/normalnn