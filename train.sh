#!/usr/bin/env bash

set -e

PROJECT_DIR=~/jupyter/satnerf
EXP_DIR=~/jupyter/satnerf/exp
EXP_NAME=debug
EPOCHS=10
#BATCH_SIZE=32768
#CHUNK=65536
#BATCH_SIZE=49152
#CHUNK=98304
#BATCH_SIZE=57344
#CHUNK=114688
BATCH_SIZE=65536
CHUNK=131072
SYSTEM_METRIC_RECORD_INTERVAL=5

if [ ! -d "$EXP_DIR/$EXP_NAME" ]; then
  mkdir "$EXP_DIR/$EXP_NAME"
fi

#dlprof --mode pytorch \
#              python3 main.py --root_dir $PROJECT_DIR/datasets/root_dir/crops_rpcs_raw/JAX_068 \
#                --img_dir $PROJECT_DIR/datasets/DFC2019/Track3-RGB-crops/JAX_068 \
#                --gt_dir $PROJECT_DIR/datasets/DFC2019/Track3-Truth \
#                --exp_name $EXP_NAME \
#                --model nerf \
#                --img_downscale 1 \
#                --cache_dir $EXP_DIR/$EXP_NAME/cache/crops_rpcs_raw/JAX_068_ds1 \
#                --logs_dir $EXP_DIR/$EXP_NAME/logs \
#                --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \
#                --gpu_id 3 \
#                --max_epochs $EPOCHS \
#                --batch_size $BATCH_SIZE \
#                --chunk $CHUNK \
#                --fc_units 256 2>> $EXP_DIR/$EXP_NAME/outputs.txt
#                --fc_units 256

(trap 'kill 0' SIGINT; python3 main.py --root_dir $PROJECT_DIR/datasets/root_dir/crops_rpcs_raw/JAX_068 \
                --img_dir $PROJECT_DIR/datasets/DFC2019/Track3-RGB-crops/JAX_068 \
                --gt_dir $PROJECT_DIR/datasets/DFC2019/Track3-Truth \
                --exp_name $EXP_NAME \
                --model nerf \
                --img_downscale 1 \
                --cache_dir $EXP_DIR/$EXP_NAME/cache/crops_rpcs_raw/JAX_068_ds1 \
                --logs_dir $EXP_DIR/$EXP_NAME/logs \
                --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \
                --gpu_id 3 \
                --max_epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --chunk $CHUNK \
                --fc_units 256 2>> $EXP_DIR/$EXP_NAME/outputs.txt & python3 capture-system-metrics.py $EXP_DIR/$EXP_NAME/sys-metrics.txt $SYSTEM_METRIC_RECORD_INTERVAL )
