#!/bin/bash
#SBATCH --job-name=HINT_dil_domainnet_15_tasks
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-7

eval "$(conda shell.bash hook)"
conda activate prompt_intact

# bash experiments/domainnet.sh
# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
# PLEASE CHANGE THIS!!!
OUTDIR=/shared/results/pkrukowski/InTactPromptCL/${DATASET}/6-task

# hard coded inputs
GPUID='0'
CONFIG=configs/dil_hint_domainnet.yaml
REPEAT=1
OVERWRITE=1

###############################################################
# ARRAY JOB PARAMETER MAPPING
###############################################################
# Define the arrays
HNET_EMBEDDING_SIZE=(48 96)
PERTURBATED_EPSILON=(0.05 0.1)
HNET_LOSS_REG=(0.01 0.1)

# Map the SLURM_ARRAY_TASK_ID (0-7) to the respective indices
idx_emb=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
idx_eps=$(( (SLURM_ARRAY_TASK_ID / 2) % 2 ))
idx_reg=$(( SLURM_ARRAY_TASK_ID % 2 ))

# Extract the specific parameter values for this job
hnet_embedding_size=${HNET_EMBEDDING_SIZE[$idx_emb]}
perturbated_epsilon=${PERTURBATED_EPSILON[$idx_eps]}
hnet_loss_reg=${HNET_LOSS_REG[$idx_reg]}

echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: Emb_Size=${hnet_embedding_size}, Eps=${perturbated_epsilon}, Reg=${hnet_loss_reg}"

###############################################################

# process inputs
mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
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