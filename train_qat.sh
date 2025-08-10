#!/usr/bin/env bash

export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

#export project_dir=/data/zis35724/jupyter/satnerf
#export exp_dir=~/jupyter/satnerf/exp-qat
#export exp_name=JAX_260_satnerf_qat
#
#if [ ! -d "$exp_dir/$exp_name" ]; then
#  mkdir "$exp_dir/$exp_name"
#fi
#
#python3 main_qat.py --root_dir $project_dir/datasets/root_dir/crops_rpcs_ba_v2/JAX_260 \
#                --img_dir $project_dir/datasets/DFC2019/Track3-RGB-crops/JAX_260 \
#                --gt_dir $project_dir/datasets/DFC2019/Track3-Truth \
#                --cache_dir $project_dir/datasets/root_dir/crops_rpcs_ba_v2/JAX_260/cache \
#                --logs_dir $exp_dir/$exp_name/logs \
#                --ckpts_dir $exp_dir/$exp_name/checkpoints \
#                --exp_name $exp_name \
#                --model sat-nerf \
#                --img_downscale 1 \
#                --gpu_id 1 \
#                --max_epochs 1 \
#                --max_train_steps 200 \
#                --batch_size 8192 \
#                --chunk 40960 \
#                --fc_units 256

python3 main_qat.py