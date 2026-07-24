#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imagenet-mix_30_tasks
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx

source activate prompt_intact_n

DATASET=ImageNet_CR
N_CLASS=200

OUTDIR=/shared/results/common/helm/IntervalActivationPromptCL/${DATASET}/30-task_dil

GPUID='0'
CONFIG=configs/imagenet-cr_prompt_dil.yaml
REPEAT=2
OVERWRITE=1

mkdir -p $OUTDIR

VAR_LOSS_SCALES=("0.1")
INTERNAL_REPRESENTATION_DRIFT_REG_SCALES=("0.0001")
FEATURE_LOSS_SCALES=("0.1")

for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    for drift in "${FEATURE_LOSS_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/coda-p/var${var}_out${out}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
          --learner_type prompt --learner_name CODAPrompt \
          --prompt_param 100 8 0.0 \
          --use_interval_activation \
          --log_dir $LOGDIR \
          --var_loss_scale $var \
          --internal_repr_drift_loss_scale $out \
          --feature_loss_scale $drift \
          --use_align_loss
    done
  done
done
