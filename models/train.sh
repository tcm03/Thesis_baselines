#!/bin/bash
PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_engcaption_cls.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val_engcaption_cls.json"

OUTPUT_DIR="./checkpoints/longvu_llama_snapugc0_txtcls1"
CKPT_NAME="longvu_llama_snapugc0_txtcls1.pt"
MODEL_PATH="./checkpoints/longvu_llama3_2"
PREV_STAGE_CHECKPOINT=""
VERSION="llama3"

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 --master_port=29503 models/train.py \
  --input_model_filename $MODEL_PATH \
  --output_model_filename $OUTPUT_DIR \
  --checkpoint_fname $CKPT_NAME \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --train_log "train_log_txtcls_2.json" \
  --train_perf_log "train_perf_txtcls_2.json" \
  --eval_perf_log "eval_perf_txtcls_2.json" \
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
  --tune_cls_head True \
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
  --eval_steps 151 \
  --save_strategy "epoch" \
  --save_steps 380 \
  --logging_steps 10 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --group_by_modality_length True \
  --resume_from_checkpoint $PREV_STAGE_CHECKPOINT
  # --dataloader_num_workers 0 \
  # --output_file "test.safetensors" \
  # --config_file "config_llama.json" \
  
