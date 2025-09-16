#!/bin/bash
#SBATCH --job-name=DualPrompt_domainnet
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx



source activate interval_activation_cl

# bash experiments/domainnet.sh
# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=/shared/results/pkrukowski/IntervalActivationPromptCL/${DATASET}/5-task

# hard coded inputs
GPUID='0'
CONFIG=configs/domainnet_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
VAR_SCALES=("0.0" "0.01" "0.1")
OUTPUT_REG_SCALES=("0.0" "1.0" "10.0")
INTERVAL_DRIFT_SCALES=("0.0" "1.0" "10.0")

for var in "${VAR_SCALES[@]}"; do
  for out in "${OUTPUT_REG_SCALES[@]}"; do
    for drift in "${INTERVAL_DRIFT_SCALES[@]}"; do
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name DualPrompt \
            --prompt_param 5 20 6 \
            --log_dir ${OUTDIR}/dual-prompt \
            --var_scale $var \
            --output_reg_scale $out \
            --interval_drift_reg_scale $drift
    done
  done
done