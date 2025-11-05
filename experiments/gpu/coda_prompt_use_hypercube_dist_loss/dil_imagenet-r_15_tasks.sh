#!/bin/bash
#SBATCH --job-name=CODA-P_dil_imagenet-r_15_tasks_Hypercube_Dist_Loss
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

# bash experiments/imagenet-r.sh
# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/15-task_use_hypercube_dist_loss

# hard coded inputs
GPUID='0'
CONFIG=configs/dil_imnet-r_prompt_15_tasks.yaml
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
VAR_SCALES=("0.1")
OUTPUT_REG_SCALES=("0.0001" "0.001" "0.01" "0.1")
INTERVAL_DRIFT_SCALES=("0.1")

for var in "${VAR_SCALES[@]}"; do
  for out in "${OUTPUT_REG_SCALES[@]}"; do
    for drift in "${INTERVAL_DRIFT_SCALES[@]}"; do
        LOGDIR=${OUTDIR}/coda-p/var${var}_out${out}_drift${drift}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
          --learner_type prompt --learner_name CODAPrompt \
          --prompt_param 100 8 0.0 \
          --use_interval_activation \
          --log_dir $LOGDIR \
          --var_scale $var \
          --output_reg_scale $out \
          --interval_drift_reg_scale $drift \
          --use_hypercube_dist_loss
    done
  done
done