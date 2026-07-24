#!/bin/bash
#SBATCH --job-name=HINT_dil_imagenet-r_15_tasks
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
OUTDIR=./${DATASET}/15-task

# hard coded inputs
GPUID='0'
CONFIG=configs/dil_hint_imnet-r_prompt_15_tasks.yaml
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
HNET_EMBEDDING_SIZE=("24 48")
PERTURBATED_EPSILON=("0.5 1.0")
HNET_LOSS_REG=("0.01 0.1")

for hnet_embedding_size in "${HNET_EMBEDDING_SIZE[@]}"; do
  for perturbated_epsilon in "${PERTURBATED_EPSILON[@]}"; do
    for hnet_loss_reg in "${HNET_LOSS_REG[@]}"; do
        LOGDIR=${OUTDIR}/hint_coda-p/embsize${hnet_embedding_size}_eps${perturb_eps}_reg${hnet_loss_reg}
        mkdir -p $LOGDIR
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
          --learner_type prompt --learner_name CODAPrompt \
          --prompt_param 100 8 0.0 \
          --use_hint \
          --log_dir $LOGDIR \
          --hnet_embedding_size $hnet_embedding_size \
          --perturbated_epsilon $perturbated_epsilon \
          --hnet_loss_reg hnet_loss_reg \
          --hnet_param 100 100
    done
  done
done