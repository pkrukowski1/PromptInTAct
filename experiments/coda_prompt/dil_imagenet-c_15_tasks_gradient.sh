#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imagenet-c_15_grad
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx

source activate prompt_intact

# experiment settings
DATASET=ImageNet_C
N_CLASS=200

# save directory
# PLEASE CHANGE THIS IF NEEDED
OUTDIR=./${DATASET}/15-task_gradient_cosine

# hard coded inputs
GPUID='0'
CONFIG=configs/imagenet-c_prompt_dil.yaml
REPEAT=1
OVERWRITE=1

###############################################################

mkdir -p $OUTDIR

CODA_P_POOL_SIZES=(196)
CODA_P_LENGTHS=(8)
VAR_LOSS_SCALE=0.01
INTERNAL_REP_DRIFT_SCALE=1.0
FEATURE_LOSS_SCALE=1.0

for pool in "${CODA_P_POOL_SIZES[@]}"; do
  for length in "${CODA_P_LENGTHS[@]}"; do
    LOGDIR=${OUTDIR}/coda-p/pool${pool}_len${length}_var${VAR_LOSS_SCALE}_out${INTERNAL_REP_DRIFT_SCALE}_drift${FEATURE_LOSS_SCALE}
    mkdir -p $LOGDIR

    python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name CODAPrompt \
      --prompt_param $pool $length 1 \
      --use_interval_activation \
      --log_dir $LOGDIR \
      --var_loss_scale $VAR_LOSS_SCALE \
      --internal_repr_drift_loss_scale $INTERNAL_REP_DRIFT_SCALE \
      --feature_loss_scale $FEATURE_LOSS_SCALE \
      --use_align_loss \
      --domain_num 15 \
      --gradient_analysis
  done
done
