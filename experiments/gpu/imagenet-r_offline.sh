#!/bin/bash
#SBATCH --job-name=Imagenet-r_offline
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

DATASET=ImageNet_R
N_CLASS=200

OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/5-task_use_hypercube_dist_loss
GPUID='0'
CONFIG=configs/imnet-r_prompt_5_tasks.yaml
REPEAT=1
OVERWRITE=0

mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type default --learner_name NormalNN --upper_bound_flag \
    --log_dir ${OUTDIR}/normalnn