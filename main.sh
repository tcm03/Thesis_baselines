#!/bin/bash
DATA_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"

torchrun --nproc_per_node=4 --master_port=29502 main.py \
  --data_paths $DATA_PATHS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --model_path $MODEL_PATH \
  --output_file "test.safetensors" \
  --config_file "config_llama.json" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8
