#!/bin/bash
DATA_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_tiny_train.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_tiny_val.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"
INPUT_CHKPT_PATH="./checkpoints/longvu_llama_snapugc0/checkpoint-epoch0.pt"
OUTPUT_CKPT_PATH="./checkpoints/longvu_llama_snapugc0"

torchrun --nproc_per_node=4 --master_port=29502 main.py \
  --input_checkpoint_path $INPUT_CHKPT_PATH \
  --output_checkpoint_path $OUTPUT_CKPT_PATH \
  --data_paths $DATA_PATHS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --model_path $MODEL_PATH \
  --output_file "test.safetensors" \
  --config_file "config_llama.json" \
  --eval_strategy "steps" \
  --save_strategy "epoch" \
  --eval_on_final_epoch True \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --learning_rate 5e-7 \
  --weight_decay 0. \
  --logging_steps 10 \
  --eval_steps 379 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  # --resume
