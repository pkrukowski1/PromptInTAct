#!/bin/bash
#SBATCH --job-name=HINT_dil_imagenet-r_15_tasks
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx4090

eval "$(conda shell.bash hook)"
conda activate prompt_intact

# experiment settings
DATASET=DIL_ImageNet_R
N_CLASS=200

# save directory
# OUTDIR=./${DATASET}/15-task
OUTDIR=/shared/results/pkrukowski/InTactPromptCL/${DATASET}/15-task/best
mkdir -p $OUTDIR

# hard coded inputs
GPUID='0'
CONFIG=configs/dil_hint_imnet-r_prompt_15_tasks.yaml
REPEAT=3
OVERWRITE=0

###############################################################
# ARRAY JOB PARAMETER MAPPING
###############################################################
# Define the arrays
hnet_embedding_size=48
perturbated_epsilon=0.1
hnet_loss_reg=0.01

echo "Parameters: Emb_Size=${hnet_embedding_size}, Eps=${perturbated_epsilon}, Reg=${hnet_loss_reg}"

###############################################################
# EXECUTION
###############################################################

LOGDIR=${OUTDIR}/hint_coda-p/embsize${hnet_embedding_size}_eps${perturbated_epsilon}_reg${hnet_loss_reg}
mkdir -p $LOGDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_hint \
  --log_dir $LOGDIR \
  --hnet_embedding_size $hnet_embedding_size \
  --perturbated_epsilon $perturbated_epsilon \
  --hnet_loss_reg $hnet_loss_reg \
  --hnet_hidden_neurons 50