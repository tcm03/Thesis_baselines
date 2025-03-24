#!/bin/bash

DATA_PATH="/raid/nthuy/SnapUGC"
TRAIN_PATH="/raid/nthuy/SnapUGC/snapugc_30s_train_short.json"
EVAL_PATH="/raid/nthuy/SnapUGC/snapugc_30s_test_short.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_endpoint=localhost:29502 \
    --max_restarts=0 \
    main.py \
    --data_path "$DATA_PATH" \
    --train_path "$TRAIN_PATH" \
    --eval_path "$EVAL_PATH" \
    --model_path "$MODEL_PATH" \
    --output_file "test.safetensors" \
    --config_file "config_llama.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \

