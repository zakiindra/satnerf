#!/usr/bin/env bash

set +e

train() {
    local PROJECT_DIR=/data/zis35724/jupyter/satnerf
    local EXP_DIR=~/jupyter/satnerf/exp-check
    local SYSTEM_METRIC_RECORD_INTERVAL=5

    local SCENE=$1
    local ALGO=$2
    local BATCH_SIZE=$3
    local MAX_TRAIN_STEPS=$4
    local FC_LAYERS=$5
    local FC_UNITS=$6
    local CHUNK=$((BATCH_SIZE * 5))
    local DOWNSCALE=1
    local NUM_GPUS=1
    local EXP_NAME=${SCENE}_ds${DOWNSCALE}_${ALGO}_bf16mixed_medium_${FC_LAYERS}_${FC_UNITS}

    if [ ! -d "$EXP_DIR/$EXP_NAME" ]; then
        mkdir -p "$EXP_DIR/$EXP_NAME"
    fi

#    (trap 'kill 0' SIGINT; python3 -W ignore:torchmetrics.rank_zero_deprecation main.py \
#        --root_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba \
#        --img_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba/crops \
#        --gt_dir $PROJECT_DIR/datasets/Track3-Truth-JAX \
#        --cache_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba/cache \
#        --exp_name $EXP_NAME \
#        --model $ALGO \
#        --img_downscale $DOWNSCALE \
#        --logs_dir $EXP_DIR/$EXP_NAME/logs \
#        --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \
#        --gpu_id $NUM_GPUS \
#        --max_train_steps $MAX_TRAIN_STEPS \
#        --batch_size $BATCH_SIZE \
#        --chunk $CHUNK \
#        --fc_layers $FC_LAYERS \
#        --fc_units $FC_UNITS 2>> $EXP_DIR/$EXP_NAME/outputs.txt & python3 capture-system-metrics.py \
#        $EXP_DIR/$EXP_NAME/sys-metrics.txt $SYSTEM_METRIC_RECORD_INTERVAL 2 $EXP_DIR/$EXP_NAME/logs )

    python3 -W ignore:torchmetrics.rank_zero_deprecation main.py \
        --root_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba \
        --img_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba/crops \
        --gt_dir $PROJECT_DIR/datasets/DFC2019/Track3-Truth-JAX \
        --cache_dir $PROJECT_DIR/datasets/Track3-preprocess/${SCENE}/ba/cache \
        --exp_name $EXP_NAME \
        --model $ALGO \
        --img_downscale $DOWNSCALE \
        --logs_dir $EXP_DIR/$EXP_NAME/logs \
        --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \
        --gpu_id $NUM_GPUS \
        --max_train_steps $MAX_TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --fc_layers $FC_LAYERS \
        --fc_units $FC_UNITS 2>> $EXP_DIR/$EXP_NAME/outputs.txt
}

train JAX_416 sat-nerf 1024 50000 8 256
train JAX_416 sat-nerf 1024 50000 8 128
train JAX_416 sat-nerf 1024 50000 8 64
train JAX_416 sat-nerf 1024 50000 6 256
train JAX_416 sat-nerf 1024 50000 6 128
train JAX_416 sat-nerf 1024 50000 6 64
train JAX_416 sat-nerf 1024 50000 4 256
train JAX_416 sat-nerf 1024 50000 4 128
train JAX_416 sat-nerf 1024 50000 4 64


