#!/bin/bash
#SBATCH --job-name=CODA-P_domainnet
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate prompt_intact

# bash experiments/domainnet.sh
# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/5-task

# hard coded inputs
GPUID='0'
CONFIG=configs/domainnet_prompt.yaml
REPEAT=1
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
var_loss_scaleS=("0.001" "0.01" "0.1" "1.0")
internal_repr_drift_loss_scaleS=("0.0001" "0.001" "0.1")
FEATURE_LOSS_SCALES=("0.0001" "0.001" "0.1")

for var in "${var_loss_scaleS[@]}"; do
  for out in "${internal_repr_drift_loss_scaleS[@]}"; do
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
