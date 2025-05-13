#!/bin/bash
PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"

PREV_STAGE_CHECKPOINT="./checkpoints/longvu_llama3_2"
OUTPUT_CHECKPOINT="./checkpoints/longvu_llama_snapugc0"

torchrun --nproc_per_node=6 --master_port=29503 main.py \
  --input_model_filename $PREV_STAGE_CHECKPOINT \
  --output_model_filename $OUTPUT_CHECKPOINT \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --model_path $MODEL_PATH \
  --output_file "test.safetensors" \
  --config_file "config_llama.json" \
  --eval_strategy "steps" \
  --save_strategy "epoch" \
  --eval_on_final_epoch True \
  --num_train_epochs 3 \
  --warmup_ratio 0.01 \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --logging_steps 5 \
  --eval_steps 253 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --group_by_modality_length True \
  --dataloader_num_workers 4 \
  # --resume_from_checkpoint $MODEL_PATH
