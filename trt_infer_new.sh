#!/usr/bin/env bash 
set +e

export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
export LD_PRELOAD=/data/anaconda3/envs/satnerf/lib/libgio-2.0.so.0

python3 eval_satnerf_trt.py
