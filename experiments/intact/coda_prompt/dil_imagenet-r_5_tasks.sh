#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imagenet-r_5_tasks_intact
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate prompt_intact

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/5-task/intact

# hard coded inputs
GPUID='0'
CONFIG=configs/intact/dil_imnet-r_prompt_5_tasks.yaml
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
LAMBDA_VAR_SCALES=("0.1")
LAMBDA_DRIFT_SCALES=("0.0001" "0.001" "0.01" "0.1")
LAMBDA_FEAT_SCALES=("0.1")

for var in "${LAMBDA_VAR_SCALES[@]}"; do
  for drift in "${LAMBDA_DRIFT_SCALES[@]}"; do
    for feat in "${LAMBDA_FEAT_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/coda-p/var${var}_drift${drift}_feat${feat}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
          --learner_type prompt --learner_name CODAPrompt \
          --prompt_param 100 8 0.0 \
          --use_intact_regularization \
          --log_dir $LOGDIR \
          --lambda_var $var \
          --lambda_drift $drift \
          --lambda_feat $feat \
          --use_align_loss
    done
  done
done