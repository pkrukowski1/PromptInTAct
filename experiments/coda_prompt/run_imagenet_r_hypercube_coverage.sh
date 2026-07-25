#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imnet-r_hypercube
#SBATCH --qos=normal
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source $HOME/miniconda3/bin/activate
conda activate /shared/results/common/miksa/envs/prompt_intact

DATASET=DIL_ImageNet_R
OUTDIR=./output/${DATASET}/15-task-coverage
CONFIG=configs/dil_imnet-r_prompt_15_tasks.yaml
REPEAT=1
OVERWRITE=0

mkdir -p $OUTDIR

LOGDIR=${OUTDIR}/coda-p/coverage
mkdir -p $LOGDIR

python -u run.py --config $CONFIG --gpuid 0 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --use_interval_activation \
    --use_align_loss \
    --use_intact_metrics \
    --log_dir $LOGDIR