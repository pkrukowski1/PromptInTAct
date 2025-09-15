# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='-1'
CONFIG=configs/cifar-100_prompt.yaml
CONFIG_FT=configs/cifar-100_ft.yaml
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
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 \
    --use_interval_activation \
    --log_dir ${OUTDIR}/dual-prompt \
    --var_scale 0.01 \
    --output_reg_scale 1.0 \
    --interval_drift_reg_scale 1.0