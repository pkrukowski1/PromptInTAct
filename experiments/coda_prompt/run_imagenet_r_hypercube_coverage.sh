#!/bin/bash
#SBATCH --job-name=CODA-P_imnet-r_hypercube
#SBATCH --qos=normal
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source $HOME/miniconda3/bin/activate
conda activate /shared/results/common/miksa/envs/prompt_intact

DATASET=ImageNet_R
OUTDIR=./output/${DATASET}/hypercube-coverage
REPEAT=1
OVERWRITE=0

declare -A SPLITS
SPLITS["5-task"]="configs/imnet-r_prompt_5_tasks.yaml"
SPLITS["10-task"]="configs/imnet-r_prompt_10_tasks.yaml"
SPLITS["20-task"]="configs/imnet-r_prompt_20_tasks.yaml"

for TASK_NAME in "${!SPLITS[@]}"; do
    CONFIG=${SPLITS[$TASK_NAME]}
    LOGDIR=${OUTDIR}/${TASK_NAME}/coda-p/coverage
    mkdir -p $LOGDIR
    python -u run.py --config $CONFIG --gpuid 0 --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name CODAPrompt \
        --prompt_param 100 8 0.0 \
        --use_interval_activation \
        --use_align_loss \
        --use_intact_metrics \
        --log_dir $LOGDIR \
        --var_loss_scale 0.1 \
        --internal_repr_drift_loss_scale 0.001 \
        --feature_loss_scale 0.1
done