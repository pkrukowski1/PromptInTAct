#!/bin/bash
#SBATCH --job-name=CODA-P_joint_ablation
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgx

source activate prompt_intact

DATASET=DIL_ImageNet_R
N_CLASS=200

OUTDIR=/shared/results/common/helm/IntervalActivationPromptCL/${DATASET}/15-task/ablation_joint

GPUID='0'
CONFIG=configs/dil_imnet-r_prompt_15_tasks.yaml
REPEAT=2
OVERWRITE=1
DATA_ROOT="/shared/sets/datasets/imagenet-r_InTAct"

mkdir -p $OUTDIR

# Optimal hyperparameters from Table 3 (ImageNet-R)
# L_Var=0.1, L_IntDrift=0.001, L_Feat=0.1, L_Align=on
#
# Table 2 already has single-term removals.
# Below: all pairwise joint removals to test redundancy vs synergy.

# --- Pair 1: w/o L_Var + L_Align (both geometric structuring terms) ---
LOGDIR=${OUTDIR}/coda-p/wo_Var_Align
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.0 \
  --internal_repr_drift_loss_scale 0.001 \
  --feature_loss_scale 0.1 \
  --data_root_dir $DATA_ROOT

# --- Pair 2: w/o L_Var + L_Feat ---
LOGDIR=${OUTDIR}/coda-p/wo_Var_Feat
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.0 \
  --internal_repr_drift_loss_scale 0.001 \
  --feature_loss_scale 0.0 \
  --use_align_loss \
  --data_root_dir $DATA_ROOT

# --- Pair 3: w/o L_Align + L_Feat ---
LOGDIR=${OUTDIR}/coda-p/wo_Align_Feat
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.1 \
  --internal_repr_drift_loss_scale 0.001 \
  --feature_loss_scale 0.0 \
  --data_root_dir $DATA_ROOT

# --- Pair 4: w/o L_IntDrift + L_Var ---
LOGDIR=${OUTDIR}/coda-p/wo_IntDrift_Var
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.0 \
  --internal_repr_drift_loss_scale 0.0 \
  --feature_loss_scale 0.1 \
  --use_align_loss \
  --data_root_dir $DATA_ROOT

# --- Pair 5: w/o L_IntDrift + L_Align ---
LOGDIR=${OUTDIR}/coda-p/wo_IntDrift_Align
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.1 \
  --internal_repr_drift_loss_scale 0.0 \
  --feature_loss_scale 0.1 \
  --data_root_dir $DATA_ROOT

# --- Pair 6: w/o L_IntDrift + L_Feat ---
LOGDIR=${OUTDIR}/coda-p/wo_IntDrift_Feat
mkdir -p $LOGDIR
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale 0.1 \
  --internal_repr_drift_loss_scale 0.0 \
  --feature_loss_scale 0.0 \
  --use_align_loss \
  --data_root_dir $DATA_ROOT
