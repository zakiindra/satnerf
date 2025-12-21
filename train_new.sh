#!/usr/bin/env bash

set +e

# export LD_LIBRARY_PATH=/data/miniconda3/envs/satnerf/lib:$LD_LIBRARY_PATH

train() {
    local PROJECT_DIR=/data/satnerf
    local EXP_DIR=/data/satnerf/exp
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
    local N_SAMPLES=72
    local EXP_NAME=${SCENE}_ds${DOWNSCALE}_${ALGO}_tmpd_${FC_LAYERS}_${FC_UNITS}_${N_SAMPLES}

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

    python3 main.py \
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
	--n_samples $N_SAMPLES \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --fc_layers $FC_LAYERS \
        --fc_units $FC_UNITS \
	1>> $EXP_DIR/$EXP_NAME/stdout.txt
	2>> $EXP_DIR/$EXP_NAME/stderr.txt
}

train JAX_068 sat-nerf 8192 2580 4 64
train JAX_068 sat-nerf 8192 2580 4 128
train JAX_068 sat-nerf 8192 2580 4 256

train JAX_068 sat-nerf 8192 2580 6 64
train JAX_068 sat-nerf 8192 2580 6 128
train JAX_068 sat-nerf 8192 2580 6 256

train JAX_068 sat-nerf 8192 2580 8 64
train JAX_068 sat-nerf 8192 2580 8 128
train JAX_068 sat-nerf 8192 2580 8 256

train JAX_068 sat-nerf 8192 2580 10 64
train JAX_068 sat-nerf 8192 2580 10 128
train JAX_068 sat-nerf 8192 2580 10 256

# for i in {1..5}; do
#     train JAX_068 sat-nerf 8192 2580 4 64
#     train JAX_068 sat-nerf 8192 2580 4 128
#     train JAX_068 sat-nerf 8192 2580 4 256
# 
#     train JAX_068 sat-nerf 8192 2580 6 64
#     train JAX_068 sat-nerf 8192 2580 6 128
#     train JAX_068 sat-nerf 8192 2580 6 256
# 
#     train JAX_068 sat-nerf 8192 2580 8 64
#     train JAX_068 sat-nerf 8192 2580 8 128
#     train JAX_068 sat-nerf 8192 2580 8 256
# 
#     train JAX_068 sat-nerf 8192 2580 10 64
#     train JAX_068 sat-nerf 8192 2580 10 128
#     train JAX_068 sat-nerf 8192 2580 10 256
# done

