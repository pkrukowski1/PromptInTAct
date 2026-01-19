#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imagenet-r_15_tasks_intactpp
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate intact_plus_plus

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=./${DATASET}/15-task/intactpp
# OUTDIR=/shared/results/pkrukowski/PromptBasedInTActPlusPlus/${DATASET}/15-task/intactpp

# hard coded inputs
GPUID='0'
CONFIG=configs/intactpp/dil_imnet-r_prompt_15_tasks.yaml
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
LAMBDA_VAR_SCALES=("0.001" "0.0001" "0.01" "0.1" "1.0")
LAMBDA_DRIFT_SCALES=("0.001" "0.0001" "0.01" "0.1" "1.0")
LAMBDA_SLOPE_SCALES=("0.5" "1.0" "2.0" "3.0")


for var in "${LAMBDA_VAR_SCALES[@]}"; do
  for drift in "${LAMBDA_DRIFT_SCALES[@]}"; do
    for slope in "${LAMBDA_SLOPE_SCALES[@]}"; do
      LOGDIR=${OUTDIR}/coda-p/var${var}_drift${drift}_slope${slope}
      mkdir -p $LOGDIR
      python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name CODAPrompt \
        --prompt_param 100 8 0.0 \
        --reg_type intactpp \
        --log_dir $LOGDIR \
        --lambda_var $var \
        --lambda_drift $drift \
        --n_last_blocks_to_finetune 0 \
        --max_slope $slope
    done
  done
done