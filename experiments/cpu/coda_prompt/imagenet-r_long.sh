# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/20-task

# hard coded inputs
GPUID='-1'
CONFIG=configs/imnet-r_prompt_long.yaml
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
VAR_SCALES=("0.001" "0.01" "0.1")
OUTPUT_REG_SCALES=("1.0" "10.0" "100.0")
INTERVAL_DRIFT_SCALES=("1.0" "10.0" "100.0")

for var in "${VAR_SCALES[@]}"; do
  for out in "${OUTPUT_REG_SCALES[@]}"; do
    for drift in "${INTERVAL_DRIFT_SCALES[@]}"; do
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name CODAPrompt \
            --prompt_param 100 8 0.0 \
            --log_dir ${OUTDIR}/coda-p \
            --var_scale $var \
            --output_reg_scale $out \
            --interval_drift_reg_scale $drift
    done
  done
done