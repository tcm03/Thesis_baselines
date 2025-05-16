#!/bin/bash
PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_engcaption.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val_engcaption.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"

PREV_STAGE_CHECKPOINT="./checkpoints/longvu_llama3_2"
OUTPUT_CHECKPOINT="./checkpoints/longvu_llama_snapugc0"
VERSION="llama3"

torchrun --nproc_per_node=2 --master_port=29503 models/train.py \
  --output_dir "/media02/nthuy/Thesis_baselines" \
  --input_model_filename $PREV_STAGE_CHECKPOINT \
  --output_model_filename $OUTPUT_CHECKPOINT \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --model_max_length 8192 \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --log_on_each_node False \
  --logging_dir /tmp/llava/test/ \
  --report_to "tensorboard" \
  --save_total_limit 1 \
  --version $VERSION \
  --mm_vision_select_layer "-2" \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --lazy_preprocess True \
  --tune_mm_mlp_adapter True \
  --tune_lm_head True \
  --freeze_mm_mlp_adapter False \
  --freeze_backbone True \
  --gradient_checkpointing True \
  --mm_projector_type sva \
  --image_token_len 144 \
  --query_num_list "[144]" \
  --lowres_token 8 \
  --video_fps 1 \
  --highres True \
  --drop_threshold 0.8 \
  --eval_strategy "steps" \
  --eval_steps 253 \
  --save_strategy "epoch" \
  --save_steps 500 \
  --logging_steps 5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --group_by_modality_length True \
  --dataloader_num_workers 0 \
  # --model_path $MODEL_PATH \
  # --output_file "test.safetensors" \
  # --config_file "config_llama.json" \
  # --resume_from_checkpoint $MODEL_PATH
